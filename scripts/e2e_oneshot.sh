#!/usr/bin/env bash
# End-to-end one-shot rental: create -> provision -> render one 58 s clip ->
# download -> destroy, with a wall-clock stamp at every boundary.
#
#   ./scripts/e2e_oneshot.sh 72636
#
# Everything measured so far has been per-render time on a box that was already
# up. That is the wrong number for the "rent it, make one video, throw it away"
# workflow, where provisioning and the cold inductor cache are paid once per
# video rather than amortised over a sweep. This script measures that number.
#
# Config note: it renders full58_sage_API.json, which is the s5 lineage —
# torch.compile on the transformer blocks only. The VAE decoder is deliberately
# NOT compiled: that is the s8/R2 change, and its 646.5 s first-run autotune
# never amortises when you generate exactly one clip.
#
# Phase stamps land in $RUN_DIR/phases.tsv as "epoch<TAB>label", so the report is
# derived from timestamps taken at the time, not from log scraping afterwards.
set -uo pipefail

MACHINE_ID="${1:?usage: e2e_oneshot.sh <machine_id>}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

RUN_DIR="${RUN_DIR:-$REPO/output/e2e_$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$RUN_DIR"
PHASES="$RUN_DIR/phases.tsv"
: > "$PHASES"

WORKFLOW=workflows/generated/full/full58_sage_API.json
AUDIO=input_files/santa_58s.mp3
IMAGE=input_files/santa-classic-portrait.png
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
stamp create_issued
say "creating from machine $MACHINE_ID"
python3 scripts/agent_vastai.py create "$MACHINE_ID" 2>&1 | tee "$RUN_DIR/create.log"
if ! grep -q "Instance created successfully" "$RUN_DIR/create.log"; then
    say "create failed - aborting before anything bills"
    exit 1
fi

# The create call does not print the id; take the newest instance we own.
for _ in $(seq 1 20); do
    INSTANCE=$(vastai show instances --raw 2>/dev/null \
        | python3 -c 'import json,sys
try: d=json.load(sys.stdin)
except Exception: d=[]
print(max((i["id"] for i in d), default=""))')
    [[ -n "$INSTANCE" ]] && break
    sleep 3
done
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
say "waiting for ssh"
for _ in $(seq 1 120); do
    url=$(vastai ssh-url "$INSTANCE" 2>/dev/null | tr -d '[:space:]')
    if [[ "$url" =~ ^ssh://root@([^:]+):([0-9]+)$ ]]; then
        SSH_HOST="${BASH_REMATCH[1]}"; SSH_PORT="${BASH_REMATCH[2]}"
        if timeout 20 ssh $(ssh_opts) "root@${SSH_HOST}" true 2>/dev/null; then
            break
        fi
    fi
    SSH_HOST=""; sleep 10
done
[[ -z "$SSH_HOST" ]] && { say "ssh never came up"; exit 1; }
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

# ---------------------------------------------------------------- upload
say "uploading workflow + assets"
rsh 'mkdir -p /workspace/scripts /workspace/results /workspace/ComfyUI/input' 60
scp $(scp_opts) -q "$AUDIO" "$IMAGE" "root@${SSH_HOST}:/workspace/ComfyUI/input/" || exit 1
scp $(scp_opts) -q "$WORKFLOW" "root@${SSH_HOST}:/workspace/full58_sage_API.json" || exit 1
scp $(scp_opts) -q scripts/run_july_tests.py "root@${SSH_HOST}:/workspace/scripts/" || exit 1
stamp uploaded

# ---------------------------------------------------------------- render
say "rendering 58 s clip (cold inductor cache)"
rsh 'nohup bash -c "while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 5; done" >/workspace/vram.log 2>&1 & echo started' 60 >/dev/null

rsh "/usr/bin/python3 /workspace/scripts/run_july_tests.py \
        --workflows /workspace/full58_sage_API.json \
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
