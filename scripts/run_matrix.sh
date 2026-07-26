#!/usr/bin/env bash
# Run a set of API workflows through run_july_tests.py, one invocation per
# workflow, and record peak VRAM alongside each timing.
#
# One invocation per workflow (rather than one call with all of them) so that a
# variant which OOMs or hangs cannot take the rest of the matrix with it, and so
# each result lands in its own file that survives a disconnect.
#
# Peak VRAM is polled from nvidia-smi rather than read from torch: the sampler
# runs inside the ComfyUI process, and on a 24 GB card whether a config fits at
# all is the actual question being measured.
#
#   ./run_matrix.sh /workspace/sweep/*.json
#
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS="${RESULTS_DIR:-/workspace/results}"
LOGS="${LOGS_DIR:-/workspace/logs}"
RUNS="${RUNS:-2}"
RUN_TIMEOUT="${RUN_TIMEOUT:-2400}"

mkdir -p "$RESULTS" "$LOGS"

restart_comfy() {
    # A variant that OOMs leaves its allocation behind: after one such failure
    # the next workflow died in 3.0s inside WanVideoModelLoader with 21.97 GiB
    # still held, and every remaining variant with it. Without this the matrix
    # reports six failures when only the first one is a real measurement.
    #
    # The inductor cache is on disk, so a restart costs a model reload, not a
    # recompile.
    supervisorctl restart comfyui >/dev/null 2>&1
    for _ in $(seq 1 60); do
        if curl -s -m 5 http://127.0.0.1:18188/system_stats >/dev/null 2>&1; then
            sleep 2
            return 0
        fi
        sleep 5
    done
    echo "    WARNING: ComfyUI did not come back after restart"
    return 1
}

poll_vram() {
    # $1 = output file. Highest reserved-MiB reading wins.
    local out="$1" peak=0 cur
    : >"$out"
    while true; do
        cur=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n1)
        if [[ ${cur:-0} -gt ${peak:-0} ]]; then
            peak=$cur
            echo "$peak" >"$out"
        fi
        sleep 2
    done
}

for wf in "$@"; do
    name=$(basename "$wf" _API.json)
    log="$LOGS/${name}.log"
    vram="$LOGS/${name}.vram"

    echo "=== $name -> $log ==="
    restart_comfy

    poll_vram "$vram" &
    poller=$!

    python3 "$HERE/run_july_tests.py" \
        --workflows "$wf" \
        --runs "$RUNS" \
        --run-timeout "$RUN_TIMEOUT" \
        --out "$RESULTS/result_${name}.json" >"$log" 2>&1
    rc=$?

    kill "$poller" 2>/dev/null
    wait "$poller" 2>/dev/null

    peak=$(cat "$vram" 2>/dev/null || echo 0)
    echo "    rc=$rc peak_vram=${peak}MiB"
    tail -n 3 "$log"
done

echo "MATRIX DONE"
