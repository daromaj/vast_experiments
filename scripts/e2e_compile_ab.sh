#!/usr/bin/env bash
# Settle whether torch.compile pays for itself on a single 58 s clip.
#
#   ./scripts/e2e_compile_ab.sh
#
# The question has been argued from numbers that never tested it: every workflow
# in the repo wires compile_args, and both variants named "nocomp" merely switch
# mode to `default`. This runs three arms on ONE box, each on a cold cache, each
# rendering the full 58 s clip:
#
#   a_autotune   max-autotune-no-cudagraphs  (ships today)
#   b_default    mode=default
#   c_nocompile  compile_args removed
#
# Arms cannot contaminate each other: inductor caches per mode, and arm C
# compiles nothing. ComfyUI is restarted between arms - on the 4090 sweep, one
# OOM left ~22 GiB allocated and every later variant died inside
# WanVideoModelLoader while still reporting a "result".
set -uo pipefail

# Ignore SIGPIPE. Piping this script to `head` killed it one line after the
# create call - past the point where an instance exists, before the deadman is
# armed and the destroy trap is installed - leaving a rental billing with
# nothing watching it. Output truncation must never be able to orphan a box.
trap '' PIPE

# No machine_id argument: the host is chosen and rented inside one search, so
# there is no window in which the chosen offer can be taken by someone else.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

RUN_DIR="${RUN_DIR:-$REPO/output/compile_ab_$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$RUN_DIR"
PHASES="$RUN_DIR/phases.tsv"
: > "$PHASES"

# Override to re-run a single arm: ARMS="b_default" ./scripts/e2e_compile_ab.sh
read -r -a ARMS <<< "${ARMS:-a_autotune b_default c_nocompile}"
AUDIO=input_files/santa_58s.mp3
IMAGE=input_files/santa-classic-portrait.png
RUN_TIMEOUT=3600
DEADLINE_S=7200

INSTANCE=""; SSH_HOST=""; SSH_PORT=""

stamp() { echo -e "$(date +%s)\t$1" | tee -a "$PHASES" >&2; }
say()   { echo "[ab $(date -u +%H:%M:%S)] $*"; }

BASE_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=20"
ssh_opts() { echo "$BASE_OPTS -p $SSH_PORT"; }
scp_opts() { echo "$BASE_OPTS -P $SSH_PORT"; }
rsh() { timeout "${2:-300}" ssh $(ssh_opts) "root@${SSH_HOST}" "$1"; }

cleanup_destroy() {
    if [[ -n "$INSTANCE" ]]; then
        say "destroying $INSTANCE"
        yes y | vastai destroy instance "$INSTANCE" 2>&1 | tail -2
        stamp destroyed
    fi
}

stamp create_issued
say "renting best-ranked cheap-egress 5090"
# Via search_cheap_egress, not agent_vastai: the latter's create re-runs the
# main search, whose inet_down >= 5000 filter excludes exactly the cheap-egress
# hosts worth renting for this.
python3 scripts/search_cheap_egress.py --max-cost-per-tb "${MAX_TB:-3.0}" \
    --create-best 2>&1 | tee "$RUN_DIR/create.log"
grep -q "Instance created successfully" "$RUN_DIR/create.log" || {
    say "create failed - nothing billed"; exit 1; }
# Guard against renting an interruptible box: a bid instance can be reclaimed
# part-way through the arms, losing the whole comparison. This already happened
# once, from a missing instance_type defaulting to 'bid'.
grep -q "Creating ON-DEMAND instance" "$RUN_DIR/create.log" || {
    say "created a BID instance - not acceptable for a multi-arm run"
    INSTANCE=$(vastai show instances --raw 2>/dev/null \
        | python3 -c 'import json,sys
try: d=json.load(sys.stdin)
except Exception: d=[]
print(max((i["id"] for i in d), default=""))')
    cleanup_destroy
    exit 1; }

for _ in $(seq 1 20); do
    INSTANCE=$(vastai show instances --raw 2>/dev/null \
        | python3 -c 'import json,sys
try: d=json.load(sys.stdin)
except Exception: d=[]
print(max((i["id"] for i in d), default=""))')
    [[ -n "$INSTANCE" ]] && break
    sleep 3
done
[[ -z "$INSTANCE" ]] && { say "no instance id"; exit 1; }
say "instance $INSTANCE"
echo "$INSTANCE" > "$RUN_DIR/instance_id"

setsid nohup "$REPO/scripts/vast_deadman.sh" "$INSTANCE" "$DEADLINE_S" \
    </dev/null >/dev/null 2>&1 &
disown 2>/dev/null || true
trap cleanup_destroy EXIT

say "waiting for ssh"
for _ in $(seq 1 120); do
    url=$(vastai ssh-url "$INSTANCE" 2>/dev/null | tr -d '[:space:]')
    if [[ "$url" =~ ^ssh://root@([^:]+):([0-9]+)$ ]]; then
        SSH_HOST="${BASH_REMATCH[1]}"; SSH_PORT="${BASH_REMATCH[2]}"
        timeout 20 ssh $(ssh_opts) "root@${SSH_HOST}" true 2>/dev/null && break
    fi
    SSH_HOST=""; sleep 10
done
[[ -z "$SSH_HOST" ]] && { say "ssh never came up"; exit 1; }
stamp ssh_up

say "waiting for provisioning + ComfyUI"
ready=0
for _ in $(seq 1 180); do
    rsh 'curl -s -m 5 http://127.0.0.1:18188/system_stats >/dev/null 2>&1 && echo UP' 60 \
        | grep -q UP && { ready=1; break; }
    sleep 10
done
[[ $ready -eq 0 ]] && { say "ComfyUI never answered"; exit 1; }
stamp comfy_up
rsh 'cat /var/log/portal/provisioning.log' 120 > "$RUN_DIR/provisioning.log" 2>&1
rsh 'grep "^environment=" /etc/supervisor/conf.d/comfyui.conf' 60 > "$RUN_DIR/wanopt_env.txt" 2>&1

say "uploading"
rsh 'mkdir -p /workspace/scripts /workspace/results /workspace/ab /workspace/ComfyUI/input' 60
scp $(scp_opts) -q "$AUDIO" "$IMAGE" "root@${SSH_HOST}:/workspace/ComfyUI/input/" || exit 1
scp $(scp_opts) -q workflows/generated/compile_ab/*_API.json "root@${SSH_HOST}:/workspace/ab/" || exit 1
scp $(scp_opts) -q scripts/run_july_tests.py "root@${SSH_HOST}:/workspace/scripts/" || exit 1
stamp uploaded

# Record the inductor cache size per arm: it is the direct evidence of how much
# each mode actually compiled, independent of wall-clock.
for arm in "${ARMS[@]}"; do
    say "arm $arm (cold cache)"
    # Wipe the inductor cache. The first version of this script did not, on the
    # assumption that inductor keys entries by mode so the arms could not share
    # them. That assumption was never verified, and it left arm B starting from
    # arm A's 360 MiB of autotuned kernels - so "cold default" could not be
    # told apart from "warm autotune". Deleting it costs seconds and removes
    # the ambiguity entirely.
    rsh 'rm -rf /tmp/torchinductor_root' 120
    rsh 'supervisorctl restart comfyui >/dev/null 2>&1; for i in $(seq 1 60); do
             curl -s -m 5 http://127.0.0.1:18188/system_stats >/dev/null 2>&1 && { sleep 2; exit 0; }
             sleep 5
         done; echo RESTART_TIMEOUT' 400
    rsh "du -sm /tmp/torchinductor_root 2>/dev/null | cut -f1" 60 \
        > "$RUN_DIR/${arm}_cache_before_mib.txt" 2>&1

    rsh "/usr/bin/python3 /workspace/scripts/run_july_tests.py \
            --workflows /workspace/ab/${arm}_API.json \
            --runs 1 --run-timeout $RUN_TIMEOUT \
            --out /workspace/results/${arm}.json 2>&1" $((RUN_TIMEOUT + 300)) \
        | tee "$RUN_DIR/${arm}.log"
    stamp "arm_${arm}_done"

    rsh "du -sm /tmp/torchinductor_root 2>/dev/null | cut -f1" 60 \
        > "$RUN_DIR/${arm}_cache_after_mib.txt" 2>&1
done

say "downloading"
mkdir -p "$RUN_DIR/videos" "$RUN_DIR/results"
rsync -az -e "ssh $(ssh_opts)" "root@${SSH_HOST}:/workspace/ComfyUI/output/" "$RUN_DIR/videos/" 2>&1 | tail -2
rsync -az -e "ssh $(ssh_opts)" "root@${SSH_HOST}:/workspace/results/" "$RUN_DIR/results/" 2>&1 | tail -2
stamp downloaded

cleanup_destroy
trap - EXIT
say "done -> $RUN_DIR"
