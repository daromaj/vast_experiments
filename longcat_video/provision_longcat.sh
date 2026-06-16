#!/bin/bash
# ==============================================================================
# Vast.ai Provisioning Script — LongCat-Video Avatar-1.5 (Audio+Image → Video)
# GPU targets: RTX 4090 (24GB), RTX 5090 (32GB)
# Uses INT8 DiT + DMD distillation (8 steps) + Cache-DIT CPU offload
# ==============================================================================

source /venv/main/bin/activate
WORKSPACE="${WORKSPACE:-/workspace}"
LONGCAT_DIR="${WORKSPACE}/LongCat-Video"
WEIGHTS_DIR="${WORKSPACE}/weights"

APT_PACKAGES=(
    aria2
    ffmpeg
    libgl1
    libglx-mesa0
    libglib2.0-0
    build-essential
    libsndfile1
)

PIP_PACKAGES=(
    "huggingface_hub[hf]"
    uv
)

function provisioning_start() {
    LOG_FILE="${WORKSPACE}/provisioning.log"
    exec > >(tee -a "$LOG_FILE") 2>&1
    echo "[$(date)] Starting LongCat-Video Avatar-1.5 provisioning..."

    # hf_transfer deprecated, not needed for hf CLI

    provisioning_print_header
    provisioning_get_apt_packages
    provisioning_clone_repo
    provisioning_install_python_deps
    provisioning_install_sageattention
    provisioning_install_cache_dit
    provisioning_download_models
    provisioning_patch_demo_script
    provisioning_print_end
}

function provisioning_print_header() {
    echo -e "\n##############################################"
    echo -e "#   LongCat-Video Avatar-1.5 Provisioning    #"
    echo -e "#     Audio + Image → Video (INT8/Distill)   #"
    echo -e "#         This will take some time           #"
    echo -e "##############################################\n"
}

function provisioning_print_end() {
    echo -e "\nProvisioning complete: Models ready.\n"
    echo "Run: ./longcat_video/run_a2v.sh --image <img> --audio <wav> --resolution 480p|720p"
}

function provisioning_get_apt_packages() {
    if [[ -n ${APT_PACKAGES[*]} ]]; then
        local apt_cmd="${APT_INSTALL:-apt-get install -y}"
        echo "[$(date)] Installing APT packages: ${APT_PACKAGES[*]}"
        sudo $apt_cmd ${APT_PACKAGES[@]}
    fi
}

function provisioning_clone_repo() {
    if [[ ! -d "$LONGCAT_DIR" ]]; then
        echo "[$(date)] Cloning LongCat-Video repository..."
        git clone --single-branch --branch main \
            https://github.com/meituan-longcat/LongCat-Video.git "$LONGCAT_DIR"
    else
        echo "[$(date)] LongCat-Video already exists, pulling latest..."
        ( cd "$LONGCAT_DIR" && git pull )
    fi
}

function provisioning_install_python_deps() {
    echo "[$(date)] Installing Python dependencies..."

    # Ensure uv is available
    pip install --no-cache-dir uv

    # Install PyTorch with CUDA support (template may not have it pre-installed)
    echo "[$(date)] Installing PyTorch (CUDA $(python3 -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'detect'))..."
    if python3 -c "import torch; exit(0 if torch.__version__ >= '2.11' else 1)" 2>/dev/null; then
        echo "PyTorch $(python3 -c 'import torch; print(torch.__version__)') already installed (>=2.11)"
    else
        echo "Installing PyTorch >= 2.11..."
        uv pip install "torch>=2.11" "torchvision>=0.20"
    fi

    # Cache-DIT requires torch >= 2.11.
    # LongCat-Video pins torch==2.6.0; we override the pin.
    echo "[$(date)] Patching requirements.txt: torch==2.6.0 → torch>=2.6.0 (for Cache-DIT)"
    sed -i 's/torch==2.6.0/torch>=2.6.0/' "${LONGCAT_DIR}/requirements.txt"

    # Core requirements from LongCat-Video
    echo "[$(date)] Installing core requirements..."
    uv pip install -r "${LONGCAT_DIR}/requirements.txt"

    # Avatar-specific requirements
    # Fix known-broken dependencies for Python 3.12 / CUDA 13
    echo "[$(date)] Patching requirements_avatar.txt: removing/updating broken deps..."
    sed -i '/libsndfile1/d' "${LONGCAT_DIR}/requirements_avatar.txt"           # system lib via apt
    sed -i '/tritonserverclient/d' "${LONGCAT_DIR}/requirements_avatar.txt"    # doesn't exist on pypi
    sed -i 's/onnxruntime==1.16.3/onnxruntime>=1.18.0/' "${LONGCAT_DIR}/requirements_avatar.txt"  # no cp312 wheel for 1.16
    echo "[$(date)] Installing avatar requirements..."
    uv pip install -r "${LONGCAT_DIR}/requirements_avatar.txt"

    # Flash Attention (critical for performance)
    echo "[$(date)] Installing flash-attn..."
    uv pip install flash-attn==2.7.4.post1 --no-build-isolation

    # Additional utilities (LongCat-Video hidden dependencies)
    uv pip install imageio[ffmpeg] tensorboard loguru ftfy einops accelerate
}

function provisioning_install_sageattention() {
    echo "[$(date)] Installing SageAttention..."

    # Try wheel first
    if uv pip install sageattention==1.0.6 --no-build-isolation 2>/dev/null; then
        echo "SageAttention installed from PyPI"
    else
        echo "Building SageAttention from source..."
        local sage_dir="${WORKSPACE}/SageAttention"
        if [[ ! -d "$sage_dir" ]]; then
            git clone https://github.com/thu-ml/SageAttention.git "$sage_dir"
        fi
        export EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=32
        ( cd "$sage_dir" && python setup.py install )
    fi

    # Check for Blackwell (5090) GPU and build sageattention3
    if nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -qi "5090"; then
        echo "[$(date)] Blackwell GPU (5090) detected — building sageattention3_blackwell..."
        local blackwell_dir="${WORKSPACE}/SageAttention/sageattention3_blackwell"
        if [[ -d "$blackwell_dir" ]]; then
            ( cd "$blackwell_dir" && python setup.py install )
            echo "sageattention3_blackwell installed"
        fi
    fi
}

function provisioning_install_cache_dit() {
    echo "[$(date)] Installing Cache-DIT (CPU offload for DiT layers)..."
    uv pip install -U cache-dit
}

function provisioning_download_models() {
    echo "[$(date)] Downloading models..."
    mkdir -p "${WEIGHTS_DIR}/LongCat-Video"
    mkdir -p "${WEIGHTS_DIR}/LongCat-Video-Avatar-1.5"

    # =========================================================================
    # Base model components: VAE, text encoder (UMT5-XXL), tokenizer
    # We only need these 3 subdirectories — skip dit/ (54 GB) not needed for Avatar
    # =========================================================================
    echo "[$(date)] Downloading VAE (508 MB)..."
    hf download meituan-longcat/LongCat-Video \
        vae/ \
        --local-dir "${WEIGHTS_DIR}/LongCat-Video" \
        --repo-type model

    echo "[$(date)] Downloading text encoder / UMT5-XXL (22.7 GB)..."
    hf download meituan-longcat/LongCat-Video \
        text_encoder/ \
        --local-dir "${WEIGHTS_DIR}/LongCat-Video" \
        --repo-type model

    echo "[$(date)] Downloading tokenizer (21 MB)..."
    hf download meituan-longcat/LongCat-Video \
        tokenizer/ \
        --local-dir "${WEIGHTS_DIR}/LongCat-Video" \
        --repo-type model

    # =========================================================================
    # Avatar-1.5 components: INT8 DiT, lora, scheduler, whisper, vocal_separator
    # =========================================================================
    echo "[$(date)] Downloading INT8 DiT model (15.9 GB)..."
    hf download meituan-longcat/LongCat-Video-Avatar-1.5 \
        base_model_int8/ \
        --local-dir "${WEIGHTS_DIR}/LongCat-Video-Avatar-1.5" \
        --repo-type model

    echo "[$(date)] Downloading DMD LoRA (2.5 GB)..."
    hf download meituan-longcat/LongCat-Video-Avatar-1.5 \
        lora/ \
        --local-dir "${WEIGHTS_DIR}/LongCat-Video-Avatar-1.5" \
        --repo-type model

    echo "[$(date)] Downloading scheduler..."
    hf download meituan-longcat/LongCat-Video-Avatar-1.5 \
        scheduler/ \
        --local-dir "${WEIGHTS_DIR}/LongCat-Video-Avatar-1.5" \
        --repo-type model

    echo "[$(date)] Downloading vocal separator (67 MB)..."
    hf download meituan-longcat/LongCat-Video-Avatar-1.5 \
        vocal_separator/ \
        --local-dir "${WEIGHTS_DIR}/LongCat-Video-Avatar-1.5" \
        --repo-type model

    # Whisper-large-v3: download only safetensors + configs, skip .bin/flax/msgpack duplicates
    # Full dir is 24.7 GB — selective download = ~3.1 GB
    echo "[$(date)] Downloading whisper-large-v3 — selective (only safetensors + configs, ~3.1 GB)..."
    hf download meituan-longcat/LongCat-Video-Avatar-1.5 \
        whisper-large-v3/ \
        --include "*.json" --include "*.txt" --include "*.safetensors" \
        --exclude "*fp32*" \
        --local-dir "${WEIGHTS_DIR}/LongCat-Video-Avatar-1.5" \
        --repo-type model

    # Root configs for Diffusers pipeline
    echo "[$(date)] Downloading pipeline configs..."
    hf download meituan-longcat/LongCat-Video-Avatar-1.5 \
        config.json model_index.json \
        --local-dir "${WEIGHTS_DIR}/LongCat-Video-Avatar-1.5" \
        --repo-type model

    echo "[$(date)] Model download complete."
    echo "Total disk usage:"
    du -sh "${WEIGHTS_DIR}/LongCat-Video" "${WEIGHTS_DIR}/LongCat-Video-Avatar-1.5"
}

function provisioning_patch_demo_script() {
    echo "[$(date)] Patching demo script for Cache-DIT integration + our paths..."

    local SCRIPT="${LONGCAT_DIR}/run_demo_avatar_single_audio_to_video.py"

    if [[ ! -f "$SCRIPT" ]]; then
        echo "WARNING: Demo script not found at $SCRIPT — skipping patch"
        return
    fi

    # Back up original
    cp "$SCRIPT" "${SCRIPT}.bak"

    python3 << 'PYEOF'
import re

script_path = "/workspace/LongCat-Video/run_demo_avatar_single_audio_to_video.py"
weights_dir = "/workspace/weights/LongCat-Video-Avatar-1.5"

with open(script_path, "r") as f:
    content = f.read()

# 1. Change default checkpoint_dir to our weights
content = content.replace(
    'default="./weights/LongCat-Video-Avatar"',
    f'default="{weights_dir}"'
)

# 2. Add cache_dit import after diffusers import (once)
if "import cache_dit" not in content:
    content = content.replace(
        "import diffusers",
        "import diffusers\nimport cache_dit"
    )

# 3. Insert Cache-DIT block before pipe.to(local_rank)
cache_dit_block = """    # Cache-DIT: bucket-style CPU offload for fitting into 24-32 GB GPUs
    print("Enabling Cache-DIT CPU offload...", flush=True)
    cache_dit.enable_cache(
        pipe,
        config={
            "offload": True,
            "offload_config": {
                "transfer_buckets": 4,
                "persistent_buckets": 64,
                "max_copy_streams": 4,
            },
        },
    )
    print("Cache-DIT enabled.", flush=True)
"""

content = content.replace(
    "    pipe.to(local_rank)",
    cache_dit_block + "\n    pipe.to(local_rank)"
)

with open(script_path, "w") as f:
    f.write(content)

print("Patched successfully")
PYEOF

    echo "Patched $SCRIPT"
    grep -n "cache_dit\|LongCat-Video-Avatar-1.5" "$SCRIPT"
}

if [[ ! -f /.noprovisioning ]]; then
    provisioning_start
fi
