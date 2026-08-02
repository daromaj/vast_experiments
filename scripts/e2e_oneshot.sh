#!/usr/bin/env bash
# End-to-end one-shot rental: create -> provision -> render one 58 s clip ->
# download -> destroy, with a wall-clock stamp at every boundary.
#
#   ./scripts/e2e_oneshot.sh best     # rank and rent in one step (preferred)
#   ./scripts/e2e_oneshot.sh 72636    # a machine_id you already chose
#
# Everything measured so far has been per-render time on a box that was already
# up. That is the wrong number for the "rent it, make one video, throw it away"
# workflow, where provisioning and the cold inductor cache are paid once per
# video rather than amortised over a sweep. This script measures that number.
#
# Config note: it renders full58_sage_API.json by default (override WORKFLOW),
# which is the s5 lineage —
# torch.compile on the transformer blocks only. The VAE decoder is deliberately
# NOT compiled: that is the s8/R2 change, and its 646.5 s first-run autotune
# never amortises when you generate exactly one clip.
#
# Phase stamps land in $RUN_DIR/phases.tsv as "epoch<TAB>label", so the report is
# derived from timestamps taken at the time, not from log scraping afterwards.
set -uo pipefail

# "best" ranks and rents in one process, which matters twice over. The
# cheap-egress hosts are the contended ones, so naming a machine_id read from an
# earlier listing loses the race often enough to be the normal case. And
# agent_vastai.py's create re-runs its OWN search, which filters
# inet_down >= 5000 - that excludes every host worth renting now that advertised
# link speed is known not to predict pull time (see PIPELINE_CEILING_MBPS in
# search_cheap_egress.py). Passing a machine_id through this script therefore
# silently bypasses the ranking that chose it.
MACHINE_ID="${1:?usage: e2e_oneshot.sh <machine_id|best>}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

RUN_DIR="${RUN_DIR:-$REPO/output/e2e_$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$RUN_DIR"
PHASES="$RUN_DIR/phases.tsv"
: > "$PHASES"

# Overridable so the same proven path can render a different arm of an A/B.
# NODE_PIN, if set, rolls ComfyUI-WanVideoWrapper back to that commit-ish after
# provisioning: a workflow from an older commit will not even load against the
# current wrapper once a node class has been renamed upstream.
WORKFLOW="${WORKFLOW:-workflows/generated/full/full58_sage_API.json}"
AUDIO="${AUDIO:-input_files/santa_58s.mp3}"
IMAGE="${IMAGE:-input_files/santa-classic-portrait.png}"
NODE_PIN="${NODE_PIN:-}"
REMOTE_WORKFLOW="/workspace/$(basename "$WORKFLOW")"
# Generous: a cold-cache 58 s render has never been timed, so the ceiling is a
# runaway guard, not an expectation.
RUN_TIMEOUT=3600
# Hard cap on the whole rental. The deadman destroys the box if this script dies.
DEADLINE_S=5400

INSTANCE=""
SSH_HOST=""
SSH_PORT=""

stamp() { echo -e "$(date +%s)\t$1" | tee -a "$PHASES" >&2; }
say()   { echo "[e2e $(date -u +%H:%M:%S)] $*"; }

# ServerAliveInterval matters: the render prints nothing for ~10 minutes, and a
# silent connection is exactly what an idle NAT timeout collects.
BASE_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=20"
ssh_opts() { echo "$BASE_OPTS -p $SSH_PORT"; }
# scp spells the port -P, not -p. Passing ssh's form makes it try to preserve
# times on a file named like the port number.
scp_opts() { echo "$BASE_OPTS -P $SSH_PORT"; }
rsh() { timeout "${2:-300}" ssh $(ssh_opts) "root@${SSH_HOST}" "$1"; }

cleanup_destroy() {
    if [[ -n "$INSTANCE" ]]; then
        say "destroying instance $INSTANCE"
        yes y | vastai destroy instance "$INSTANCE" 2>&1 | tail -2
        stamp destroyed
    fi
}

# ---------------------------------------------------------------- create
# create + id-resolution is serialised under a lock. The vast CLI does not
# return the new instance id, so it has to be inferred from the account's
# instance list - and two copies of this script running in parallel (the two
# arms of an A/B) would otherwise each claim whichever instance appeared last,
# which is a coin flip over whose box is whose.
LOCK="${TMPDIR:-/tmp}/e2e_oneshot_create.lock"
exec 200>"$LOCK"
flock -x -w 900 200 || { say "timed out waiting for the create lock"; exit 1; }

list_instance_ids() {
    vastai show instances --raw 2>/dev/null \
        | python3 -c 'import json,sys
try: d=json.load(sys.stdin)
except Exception: d=[]
print(" ".join(str(i["id"]) for i in d))'
}

BEFORE=" $(list_instance_ids) "
stamp create_issued
if [[ "$MACHINE_ID" == "best" ]]; then
    say "ranking and renting the best cheap-bandwidth 5090 in one step"
    # SKIP carries machine_ids that already failed this attempt, so a retry
    # cannot be handed the same broken host again.
    python3 scripts/search_cheap_egress.py --max-cost-per-tb "${MAX_TB:-3.0}" \
        ${SKIP:+--skip $SKIP} --create-best 2>&1 | tee "$RUN_DIR/create.log"
else
    say "creating from machine $MACHINE_ID"
    python3 scripts/agent_vastai.py create "$MACHINE_ID" 2>&1 | tee "$RUN_DIR/create.log"
fi
if ! grep -q "Instance created successfully" "$RUN_DIR/create.log"; then
    say "create failed - aborting before anything bills"
    flock -u 200
    exit 1
fi
# A bid instance can be reclaimed mid-render, losing the rental and the clip.
# This has already happened once, from a missing instance_type defaulting to
# 'bid'. Only the ranked path prints this line, so only it is checked.
if [[ "$MACHINE_ID" == "best" ]] \
   && ! grep -q "Creating ON-DEMAND instance" "$RUN_DIR/create.log"; then
    say "created a BID instance - not acceptable for a one-shot render"
    for id in $(list_instance_ids); do
        [[ "$BEFORE" == *" $id "* ]] || INSTANCE="$id"
    done
    flock -u 200
    cleanup_destroy
    exit 1
fi

# Identify by set difference against the pre-create snapshot rather than by
# "highest id", so an instance belonging to the other arm can never be adopted.
for _ in $(seq 1 20); do
    for id in $(list_instance_ids); do
        [[ "$BEFORE" == *" $id "* ]] || { INSTANCE="$id"; break; }
    done
    [[ -n "$INSTANCE" ]] && break
    sleep 3
done
flock -u 200
[[ -z "$INSTANCE" ]] && { say "could not resolve instance id"; exit 1; }
say "instance $INSTANCE"
echo "$INSTANCE" > "$RUN_DIR/instance_id"

# Arm the deadman before anything can go wrong. An orphaned rental costs more
# than the whole experiment.
setsid nohup "$REPO/scripts/vast_deadman.sh" "$INSTANCE" "$DEADLINE_S" \
    </dev/null >/dev/null 2>&1 &
disown 2>/dev/null || true

trap cleanup_destroy EXIT

# ------------------------------------------------------------------ boot
# A host that cannot start the container never opens ssh, and waiting for a
# timeout to notice costs the full 20 minutes of billing for nothing. Vast
# already knows - it puts the reason in status_msg - so ASK instead of waiting.
#
# Seen 2026-08-02 on machine 139787: instance stuck in `loading` with
#   "Error response from daemon: failed to resolve reference
#    docker.io/vastai/comfy:v0.28.0-cuda-12.9-py312: not found"
# while that tag was demonstrably present on Docker Hub (published 2026-07-16).
# A broken or rate-limited registry mirror on the host, not a stale pin - which
# is exactly the distinction this has to surface, because bumping the image tag
# invalidates the cached SageAttention wheel and would have been the wrong fix.
#
# The classifier itself is scripts/instance_fault.py, NOT an inline heredoc.
# It was inline once; the Python contained single-quoted strings, which closed
# the shell's own single-quoted argument early, and python3 answered SyntaxError
# on every poll for a whole rental without anyone noticing. Keep it in a file.
#
# The raw payload is teed to the run dir and probe errors are kept, so that a
# host which fails in a way this does NOT recognise is still diagnosable after
# the instance is destroyed. That is the expensive part to lose.
instance_fault() {
    vastai show instance "$INSTANCE" --raw 2>/dev/null \
        | tee "$RUN_DIR/instance_status.json" \
        | python3 "$REPO/scripts/instance_fault.py" 2>>"$RUN_DIR/fault_probe.err"
}

say "waiting for ssh"
for i in $(seq 1 120); do
    url=$(vastai ssh-url "$INSTANCE" 2>/dev/null | tr -d '[:space:]')
    if [[ "$url" =~ ^ssh://root@([^:]+):([0-9]+)$ ]]; then
        SSH_HOST="${BASH_REMATCH[1]}"; SSH_PORT="${BASH_REMATCH[2]}"
        if timeout 20 ssh $(ssh_opts) "root@${SSH_HOST}" true 2>/dev/null; then
            break
        fi
    fi
    SSH_HOST=""
    # Not every iteration: this is an API call, and the failure it looks for
    # takes a little while to surface. From the 3rd, every 3rd.
    if [[ $i -ge 3 && $(( i % 3 )) -eq 0 ]]; then
        fault=$(instance_fault)
        if [[ -n "$fault" ]]; then
            say "HOST FAULT: $fault"
            say "this host cannot start the container - abandoning it rather "
            say "than waiting out the ssh timeout"
            stamp host_fault
            echo "$fault" > "$RUN_DIR/host_fault.txt"
            # 75 = EX_TEMPFAIL: retry on another host, do not treat as a bug.
            exit 75
        fi
    fi
    sleep 10
done
if [[ -z "$SSH_HOST" ]]; then
    say "ssh never came up"
    # Say WHY, as far as vast will admit to. Without this the instance is
    # destroyed and the only record of a 20-minute failure is "ssh never came
    # up", which is a symptom and not a cause.
    say "last known status: $(python3 "$REPO/scripts/instance_fault.py" \
        --describe < "$RUN_DIR/instance_status.json" 2>/dev/null \
        || echo 'no status captured')"
    [[ -s "$RUN_DIR/fault_probe.err" ]] && \
        say "fault probe errored: $(head -c 200 "$RUN_DIR/fault_probe.err")"
    exit 1
fi
stamp ssh_up
say "ssh root@${SSH_HOST}:${SSH_PORT}"

# ------------------------------------------------------------ provisioning
# "Queueable" is the real gate: the provisioning script has finished AND ComfyUI
# answers. Checking only the log would start uploading into a box whose ComfyUI
# is still booting.
say "waiting for provisioning + ComfyUI"
ready=0
for _ in $(seq 1 180); do
    if rsh 'curl -s -m 5 http://127.0.0.1:18188/system_stats >/dev/null 2>&1 && echo UP' 60 \
        | grep -q UP; then
        ready=1; break
    fi
    sleep 10
done
[[ $ready -eq 0 ]] && { say "ComfyUI never answered"; rsh 'tail -40 /var/log/portal/provisioning.log' 60 > "$RUN_DIR/provisioning_tail.log"; exit 1; }
stamp comfy_up

rsh 'cat /var/log/portal/provisioning.log' 120 > "$RUN_DIR/provisioning.log" 2>&1
rsh 'grep -c "y-encode cache" /var/log/portal/*.log 2>/dev/null; grep "^environment=" /etc/supervisor/conf.d/comfyui.conf' 60 \
    > "$RUN_DIR/wanopt_env.txt" 2>&1

# ------------------------------------------------------------- node pin
# Rolling the wrapper back also reverts the WANOPT multitalk patch, since that
# is an edit to a tracked file - so R3/R6 vanish and their env flags go inert.
# No separate un-patch step is needed, but record the resulting HEAD so the run
# is auditable.
if [[ -n "$NODE_PIN" ]]; then
    say "pinning ComfyUI-WanVideoWrapper to $NODE_PIN"
    # The directory is globbed, not spelled out: povision_fp8.sh clones from a
    # .git URL and git keeps that in the basename, so this repo lands in
    # custom_nodes/ComfyUI-WanVideoWrapper.git/ while its siblings do not have
    # the suffix. Hardcoding the un-suffixed name cost one rental.
    #
    # Note the fetch has no --depth: provisioning clones full, and passing a
    # depth to fetch would truncate that history and could drop the very commit
    # being pinned to.
    rsh "cd \$(ls -d /workspace/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper*/ | head -1) \
         && { git checkout -f $NODE_PIN 2>/dev/null || { git fetch origin 2>&1 | tail -1; git checkout -f $NODE_PIN; } ; } 2>&1 | tail -3 \
         && git log -1 --format='PINNED %h %ad %s' --date=short" 300 \
        | tee "$RUN_DIR/node_pin.log"
    grep -q "^PINNED" "$RUN_DIR/node_pin.log" || { say "pin failed"; exit 1; }

    supervisor_restart_and_wait() {
        rsh 'supervisorctl restart comfyui' 120 >/dev/null
        for _ in $(seq 1 60); do
            rsh 'curl -s -m 5 http://127.0.0.1:18188/system_stats >/dev/null 2>&1 && echo UP' 60 \
                | grep -q UP && return 0
            sleep 5
        done
        return 1
    }
    # The pinned node's class list is the actual test: a wrapper that fails to
    # import still leaves ComfyUI answering /system_stats perfectly happily.
    node_present() {
        rsh "curl -s -m 20 http://127.0.0.1:18188/object_info/$1 | head -c 200" 60 \
            | grep -q "$1"
    }

    supervisor_restart_and_wait || { say "ComfyUI did not come back after pin"; exit 1; }
    if ! node_present WanVideoSampler; then
        # Only now install the pinned revision's requirements. Doing it
        # unconditionally risks downgrading a package the current torch or the
        # SageAttention wheel depends on, to fix a problem that usually is not
        # there - January needs older deps, and older deps are already satisfied.
        say "wrapper did not load; installing its pinned requirements"
        rsh 'source /venv/main/bin/activate \
             && pip install --no-cache-dir -r $(ls -d /workspace/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper*/ | head -1)requirements.txt 2>&1 | tail -20' 600 \
            | tee "$RUN_DIR/node_pin_pip.log"
        supervisor_restart_and_wait || { say "ComfyUI did not come back after pip"; exit 1; }
        node_present WanVideoSampler || { say "pinned wrapper will not load - aborting"; exit 1; }
    fi
    rsh 'curl -s -m 20 http://127.0.0.1:18188/object_info | python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d),\"node classes\"); print([k for k in d if \"Wav2Vec\" in k])"' 90 \
        >> "$RUN_DIR/node_pin.log" 2>&1
    stamp node_pinned
fi

# ---------------------------------------------------------------- upload
say "uploading workflow + assets"
rsh 'mkdir -p /workspace/scripts /workspace/results /workspace/ComfyUI/input' 60
scp $(scp_opts) -q "$AUDIO" "$IMAGE" "root@${SSH_HOST}:/workspace/ComfyUI/input/" || exit 1
scp $(scp_opts) -q "$WORKFLOW" "root@${SSH_HOST}:${REMOTE_WORKFLOW}" || exit 1
scp $(scp_opts) -q scripts/run_july_tests.py "root@${SSH_HOST}:/workspace/scripts/" || exit 1
stamp uploaded

# ---------------------------------------------------------------- render
say "rendering 58 s clip (cold inductor cache)"
rsh 'nohup bash -c "while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 5; done" >/workspace/vram.log 2>&1 & echo started' 60 >/dev/null

rsh "/usr/bin/python3 /workspace/scripts/run_july_tests.py \
        --workflows ${REMOTE_WORKFLOW} \
        --runs 1 --run-timeout $RUN_TIMEOUT \
        --out /workspace/results/e2e_full58.json 2>&1" $((RUN_TIMEOUT + 300)) \
    | tee "$RUN_DIR/render.log"
stamp rendered

# Bracket the pattern so it cannot match the shell running it - an unbracketed
# pkill -f kills its own command line first and returns nothing.
rsh 'pkill -f "nvidia-s[m]i --query-gpu=memory.used" ; sort -n /workspace/vram.log | tail -1' 60 \
    > "$RUN_DIR/peak_vram_mib.txt" 2>&1

# -------------------------------------------------------------- download
say "downloading outputs"
mkdir -p "$RUN_DIR/videos" "$RUN_DIR/results"
rsync -az --info=stats1 -e "ssh $(ssh_opts)" \
    "root@${SSH_HOST}:/workspace/ComfyUI/output/" "$RUN_DIR/videos/" 2>&1 | tail -3
rsync -az -e "ssh $(ssh_opts)" \
    "root@${SSH_HOST}:/workspace/results/" "$RUN_DIR/results/" 2>&1 | tail -2
stamp downloaded

# --------------------------------------------------------------- destroy
cleanup_destroy
trap - EXIT

say "done -> $RUN_DIR"
python3 scripts/e2e_report.py "$RUN_DIR" | tee "$RUN_DIR/report.md"
