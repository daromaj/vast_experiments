#!/usr/bin/env bash
# Quality A/B: the January 2026 InfiniteTalk stack against the current optimised
# one, on two 5090 rentals in parallel, one 58 s clip each.
#
#   ./scripts/quality_ab_jan_vs_now.sh              # auto-pick the top two hosts
#   ./scripts/quality_ab_jan_vs_now.sh 141317 110750
#
# Arm A pins ComfyUI-WanVideoWrapper back to 339e0fe (2026-01-23). That is not
# optional: node 137's class was renamed upstream (DownloadAndLoadWav2VecModel ->
# Wav2VecModelLoader), so the January workflow will not load against the current
# wrapper at all. The checkout also reverts the WANOPT multitalk patch, so the
# January arm gets none of the R3/R6 speedups either.
#
# Two boxes rather than one because the two arms need different wrapper
# checkouts, and swapping between them in place risks their pip requirements
# colliding. The cost is that render wall-clock is host-confounded and should not
# be quoted as a clean speed comparison; the speed numbers already exist in
# july_test.md. This run is about what the frames look like.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
AB_DIR="$REPO/output/jan_vs_now_$TS"
JAN_PIN="${JAN_PIN:-339e0fe}"

mkdir -p "$AB_DIR"
say() { echo "[ab $(date -u +%H:%M:%S)] $*"; }

# ------------------------------------------------------------ build arms
say "building both arms"
python3 scripts/build_quality_ab.py | tee "$AB_DIR/build.log" || exit 1

# ----------------------------------------------------------- pick hosts
MACHINE_A="${1:-}"
MACHINE_B="${2:-}"
if [[ -z "$MACHINE_A" || -z "$MACHINE_B" ]]; then
    say "picking the two best-ranked 5090 hosts"
    python3 scripts/search_cheap_egress.py --limit 10 2>&1 | tee "$AB_DIR/search.log"
    # Data rows are the ones starting with a machine id; the header and the
    # criteria preamble are not.
    mapfile -t PICKS < <(grep -E '^\s+[0-9]+\s' "$AB_DIR/search.log" | awk '{print $1}')
    [[ ${#PICKS[@]} -lt 2 ]] && { say "fewer than two hosts passed the filters"; exit 1; }
    MACHINE_A="${PICKS[0]}"
    MACHINE_B="${PICKS[1]}"
fi
say "arm A (january) -> machine $MACHINE_A"
say "arm B (current) -> machine $MACHINE_B"

# --------------------------------------------------------------- render
# e2e_oneshot.sh serialises its own create + instance-id resolution under a
# lock, so launching both at once cannot mix the two instances up.
RUN_DIR="$AB_DIR/A_january" \
WORKFLOW=workflows/generated/quality_ab/A_january_API.json \
NODE_PIN="$JAN_PIN" \
    ./scripts/e2e_oneshot.sh "$MACHINE_A" > "$AB_DIR/A_january.log" 2>&1 &
PID_A=$!

RUN_DIR="$AB_DIR/B_current" \
WORKFLOW=workflows/generated/quality_ab/B_current_API.json \
    ./scripts/e2e_oneshot.sh "$MACHINE_B" > "$AB_DIR/B_current.log" 2>&1 &
PID_B=$!

say "arm A pid $PID_A, arm B pid $PID_B - tail $AB_DIR/{A_january,B_current}.log"
wait $PID_A; RC_A=$?
wait $PID_B; RC_B=$?
say "arm A exit $RC_A, arm B exit $RC_B"

# -------------------------------------------------------------- collect
# One flat directory of unmistakably named files, because the whole point is to
# put the two clips side by side later. The -audio variant is the real output;
# VHS_VideoCombine also writes a silent one, kept but marked.
say "collecting videos"
VID="$AB_DIR/videos"
mkdir -p "$VID"
collect() {
    local arm_dir="$1" name="$2" found=0
    for f in "$arm_dir"/videos/*-audio.mp4; do
        [[ -e "$f" ]] || continue
        cp "$f" "$VID/${name}.mp4"; found=1
    done
    for f in "$arm_dir"/videos/*.mp4; do
        [[ -e "$f" && "$f" != *-audio.mp4 ]] || continue
        cp "$f" "$VID/${name}_silent.mp4"
    done
    [[ $found -eq 1 ]] || echo "WARNING: no -audio.mp4 for $name"
}
collect "$AB_DIR/A_january" "A_january2026_6step_dpmpp_sde"
collect "$AB_DIR/B_current"  "B_current_4step_flowmatch_distill"
ls -la "$VID" | tee "$AB_DIR/videos.txt"

# ------------------------------------------------------------- measure
A="$VID/A_january2026_6step_dpmpp_sde.mp4"
B="$VID/B_current_4step_flowmatch_distill.mp4"
if [[ -e "$A" && -e "$B" ]]; then
    say "metrics"
    python3 scripts/quality_metrics.py \
        "january_6step=$A" "current_4step=$B" 2>&1 | tee "$AB_DIR/quality_metrics.txt"
    say "frame strips + difference maps"
    # Spread over the whole 58 s, not just the opening: window seams land every
    # ~3 s and any lip-sync drift shows up late, not early.
    python3 scripts/compare_quality.py --out "$AB_DIR/frames" \
        --timestamps 00:00:01,00:00:05,00:00:15,00:00:30,00:00:45,00:00:57 \
        "january_6step=$A" "current_4step=$B" 2>&1 | tee "$AB_DIR/compare_quality.txt"
    for pair in "january:$A" "current:$B"; do
        say "smoothness ${pair%%:*}"
        python3 scripts/check_smoothness.py "${pair#*:}" 2>&1 \
            | tee "$AB_DIR/smoothness_${pair%%:*}.txt"
    done
else
    say "one or both videos missing - skipping metrics"
fi

say "done -> $AB_DIR"
[[ $RC_A -eq 0 && $RC_B -eq 0 ]]
