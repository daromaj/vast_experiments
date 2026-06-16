#!/bin/bash
# ==============================================================================
# LongCat-Video Avatar-1.5: Audio + Image → Video Inference
#
# Usage:
#   ./run_a2v.sh --image <path> --audio <path> [--resolution 480p|720p] [--prompt "..."]
#   ./run_a2v.sh --image cat.jpg --audio speech.wav --resolution 720p
#   ./run_a2v.sh --image cat.jpg --audio speech.wav --resolution 480p \
#                --prompt "A person speaking on stage, dramatic lighting"
# ==============================================================================

set -euo pipefail

# --- Defaults ---
RESOLUTION="480p"
PROMPT="A person speaking, facing the camera. Natural lighting, high quality video."
OUTPUT_DIR="./outputs_avatar"
NUM_SEGMENTS="auto"            # auto = calculated from audio length; override with --num-segments N
STAGE="ai2v"                # ai2v = audio+image to video; at2v = audio+text to video
GPU_COUNT="${GPU_COUNT:-1}"

# Paths (override via env or auto-detect)
WORKSPACE="${WORKSPACE:-/workspace}"
LONGCAT_DIR="${LONGCAT_DIR:-${WORKSPACE}/LongCat-Video}"
WEIGHTS_DIR="${WEIGHTS_DIR:-${WORKSPACE}/weights}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${WEIGHTS_DIR}/LongCat-Video-Avatar-1.5}"

# --- Parse arguments ---
IMAGE=""
AUDIO=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --image) IMAGE="$2"; shift 2 ;;
        --audio) AUDIO="$2"; shift 2 ;;
        --resolution) RESOLUTION="$2"; shift 2 ;;
        --prompt) PROMPT="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --stage) STAGE="$2"; shift 2 ;;
        --num-segments) NUM_SEGMENTS="$2"; shift 2 ;;
        --gpus) GPU_COUNT="$2"; shift 2 ;;
        --checkpoint-dir) CHECKPOINT_DIR="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 --image <path> --audio <path> [--resolution 480p|720p] [--prompt ...]"
            exit 1
            ;;
    esac
done

# --- Validate ---
if [[ -z "$IMAGE" ]]; then
    echo "ERROR: --image is required"
    exit 1
fi
if [[ -z "$AUDIO" ]]; then
    echo "ERROR: --audio is required"
    exit 1
fi
if [[ ! -f "$IMAGE" ]]; then
    echo "ERROR: Image file not found: $IMAGE"
    exit 1
fi
if [[ ! -f "$AUDIO" ]]; then
    echo "ERROR: Audio file not found: $AUDIO"
    exit 1
fi
if [[ "$RESOLUTION" != "480p" && "$RESOLUTION" != "720p" ]]; then
    echo "ERROR: --resolution must be 480p or 720p, got: $RESOLUTION"
    exit 1
fi

# Resolve absolute paths
IMAGE="$(realpath "$IMAGE")"
AUDIO="$(realpath "$AUDIO")"

# --- Auto-calculate num_segments from audio duration ---
# 480p: 93 frames @ 25fps, 13 cond frames → 3.72s first seg + 3.2s each extra
# 720p: 93 frames @ 16fps, 13 cond frames → 5.81s first seg + 5.0s each extra
if [[ "$NUM_SEGMENTS" == "auto" || "$NUM_SEGMENTS" -eq 0 ]]; then
    AUDIO_DURATION=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$AUDIO" 2>/dev/null || echo "0")
    if [[ "$AUDIO_DURATION" != "0" ]]; then
        if [[ "$RESOLUTION" == "480p" ]]; then
            NUM_FRAMES=93; FPS=25
        else
            NUM_FRAMES=93; FPS=16
        fi
        COND_FRAMES=13
        NEEDED=$(python3 -c "
dur = float('$AUDIO_DURATION')
fps = $FPS
nf = $NUM_FRAMES
cf = $COND_FRAMES
first = nf / fps
extra = (nf - cf) / fps
n = max(1, int((dur - first) / extra + 1.5))
print(n)
")
        if [[ "$NEEDED" -lt 1 ]]; then NEEDED=1; fi
        NUM_SEGMENTS="$NEEDED"
        echo "Audio: ${AUDIO_DURATION}s → auto segments: $NUM_SEGMENTS"
    fi
fi

# --- Activate env if on Vast.ai ---
if [[ -f /venv/main/bin/activate ]]; then
    source /venv/main/bin/activate
fi

# --- Create input JSON ---
INPUT_JSON="/tmp/longcat_a2v_input_$$.json"
cat > "$INPUT_JSON" << EOF
{
    "prompt": "${PROMPT}",
    "cond_image": "${IMAGE}",
    "cond_audio": {
        "person1": "${AUDIO}"
    }
}
EOF

echo "=== LongCat-Video Avatar-1.5 ==="
echo "  Image:      $IMAGE"
echo "  Audio:      $AUDIO"
echo "  Resolution: $RESOLUTION"
echo "  Stage:      $STAGE"
echo "  GPUs:       $GPU_COUNT"
echo "  Prompt:     $PROMPT"
echo "  Output:     $OUTPUT_DIR"
echo "  Checkpoint: $CHECKPOINT_DIR"
echo "  JSON:       $INPUT_JSON"
echo ""

# --- Build torchrun command ---
TORCHRUN_ARGS=(
    --standalone
    --nnodes=1
    --nproc_per_node="$GPU_COUNT"
)

PYTHON_ARGS=(
    "${LONGCAT_DIR}/run_demo_avatar_single_audio_to_video.py"
    --input_json "$INPUT_JSON"
    --output_dir "$OUTPUT_DIR"
    --resolution "$RESOLUTION"
    --stage_1 "$STAGE"
    --num_segments "$NUM_SEGMENTS"
    --checkpoint_dir "$CHECKPOINT_DIR"
    --model_type "avatar-v1.5"
    --use_int8
    --use_distill
)

# Multi-GPU: enable context parallelism (splits video sequence across GPUs)
if [[ "$GPU_COUNT" -gt 1 ]]; then
    PYTHON_ARGS+=(--context_parallel_size "$GPU_COUNT")
    echo "  CP:          context_parallel_size=$GPU_COUNT (sequence sharded across GPUs)"
fi

echo "Running: torchrun ${TORCHRUN_ARGS[*]} ${PYTHON_ARGS[*]}"
echo ""

cd "$LONGCAT_DIR"
torchrun "${TORCHRUN_ARGS[@]}" "${PYTHON_ARGS[@]}"

EXIT_CODE=$?

# --- Cleanup ---
rm -f "$INPUT_JSON"

if [[ $EXIT_CODE -eq 0 ]]; then
    echo ""
    echo "Done! Output saved to: $OUTPUT_DIR"
    ls -lh "$OUTPUT_DIR"/
else
    echo "ERROR: Inference failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi
