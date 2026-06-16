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
    uv
)

function provisioning_start() {
    LOG_FILE="${WORKSPACE}/provisioning.log"
    exec > >(tee -a "$LOG_FILE") 2>&1
    echo "[$(date)] Starting LongCat-Video Avatar-1.5 provisioning..."

    provisioning_print_header
    provisioning_get_apt_packages
    provisioning_clone_repo
    provisioning_install_python_deps
    provisioning_install_sageattention
    provisioning_install_cache_dit
    provisioning_download_models
    provisioning_enable_sage_attn_config
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

    # Flash Attention (optional — SageAttention/flashattn3 is the primary path)
    # Skip if building takes too long; inference works with SageAttention
    echo "[$(date)] Skipping flash-attn build (use SageAttention via flashattn3 instead)"
    # uv pip install flash-attn==2.7.4.post1 --no-build-isolation

    # Additional utilities (LongCat-Video hidden dependencies)
    uv pip install imageio[ffmpeg] tensorboard loguru ftfy einops accelerate

    # Safeguard: ensure torch wasn't downgraded by any dependency resolver
    echo "[$(date)] Verifying PyTorch >= 2.11 (Cache-DIT requirement)..."
    python3 -c "
import torch
v = torch.__version__
if v < '2.11':
    print(f'Torch {v} too old — upgrading...')
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'uv', 'pip', 'install', 'torch>=2.11', 'torchvision>=0.20'])
else:
    print(f'Torch {v} OK')
"
}

function provisioning_install_sageattention() {
    echo "[$(date)] Installing SageAttention..."

    local wheel_dir="${WORKSPACE}/sage_wheels"
    mkdir -p "$wheel_dir"

    # Pre-built wheels hosted on GitHub — no compilation needed
    # 5090 (Blackwell): sageattn3 wheel
    # 4090 (Ada): sageattention 2.2.0 4090 wheel
    # Default: sageattention 2.2.0 generic wheel

    local wheels=()

    if nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -qi "5090"; then
        echo "[$(date)] RTX 5090 detected — using sageattn3 wheel..."
        wheels+=("https://github.com/daromaj/vast_experiments/raw/master/python/sageattn3-1.0.0-cp312-cp312-linux_x86_64.whl")
    fi

    # Always install base sageattention 2.2.0 (provides flash_attn_interface for flashattn3)
    if nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -qi "4090"; then
        wheels+=("https://github.com/daromaj/vast_experiments/raw/master/python/sageattention-2.2.0-cp312-cp312-linux_x86_64_4090.whl")
    else
        wheels+=("https://github.com/daromaj/vast_experiments/raw/master/python/sageattention-2.2.0-cp312-cp312-linux_x86_64.whl")
    fi

    for url in "${wheels[@]}"; do
        echo "[$(date)] Downloading: $url"
        provisioning_download "$url" "$wheel_dir"
    done

    echo "[$(date)] Installing SageAttention wheels..."
    pip install --no-cache-dir "$wheel_dir"/*.whl 2>&1 | tail -5

    echo "[$(date)] SageAttention installed"
}

function provisioning_download() {
    local url="$1"
    local dir="$2"
    local filename="$3"
    local auth_header=""

    if [[ -z "$filename" ]]; then
        filename=$(basename "${url%%\?*}")
    fi

    if [[ -f "${dir}/${filename}" ]]; then
        echo "  Already downloaded: $filename"
        return
    fi

    # Detect HuggingFace URLs and add auth if token exists
    if [[ -n $HF_TOKEN && $url =~ ^https://([a-zA-Z0-9_-]+\\.)?huggingface\\.co(/|$|\\?) ]]; then
        auth_header="--header=Authorization: Bearer $HF_TOKEN"
    fi

    if [[ -n $auth_header ]]; then
        aria2c -x 16 -s 16 -k 1M -c --summary-interval=0 --console-log-level=warn \
            --allow-overwrite=true --auto-file-renaming=false --file-allocation=none \
            -o "$filename" $auth_header -d "$dir" "$url"
    else
        aria2c -x 16 -s 16 -k 1M -c --summary-interval=0 --console-log-level=warn \
            --allow-overwrite=true --auto-file-renaming=false --file-allocation=none \
            -o "$filename" -d "$dir" "$url"
    fi
}

function provisioning_install_cache_dit() {
    echo "[$(date)] Installing Cache-DIT (CPU offload for DiT layers)..."
    uv pip install -U cache-dit
}

function provisioning_enable_sage_attn_config() {
    # Enable flashattn3 (SageAttention) in the INT8 DiT config
    # LongCat-Video uses flash_attn_interface from SageAttention for flashattn3
    local CONFIG="${WEIGHTS_DIR}/LongCat-Video-Avatar-1.5/base_model_int8/config.json"
    if [[ -f "$CONFIG" ]]; then
        echo "[$(date)] Enabling flashattn3 (SageAttention) in DiT config..."
        python3 -c "
import json
with open('$CONFIG', 'r') as f:
    c = json.load(f)
c['enable_flashattn3'] = True
c['enable_flashattn2'] = False
with open('$CONFIG', 'w') as f:
    json.dump(c, f, indent=2)
print('flashattn3 enabled, flashattn2 disabled')
"
    fi
}

function provisioning_download_models() {
    echo "[$(date)] Downloading models (aria2c -x16 for all files)..."
    mkdir -p "${WEIGHTS_DIR}/LongCat-Video"/{vae,text_encoder,tokenizer}
    mkdir -p "${WEIGHTS_DIR}/LongCat-Video-Avatar-1.5"/{base_model_int8,lora,scheduler,vocal_separator,whisper-large-v3}

    HF_BASE="https://huggingface.co"

    # ===== LongCat-Video base =====
    echo "[$(date)] VAE (508 MB)..."
    provisioning_download "$HF_BASE/meituan-longcat/LongCat-Video/resolve/main/vae/diffusion_pytorch_model.safetensors" "${WEIGHTS_DIR}/LongCat-Video/vae"
    provisioning_download "$HF_BASE/meituan-longcat/LongCat-Video/resolve/main/vae/config.json" "${WEIGHTS_DIR}/LongCat-Video/vae"

    echo "[$(date)] Text encoder / UMT5-XXL (5 shards, 9.6 GB)..."
    for i in 01 02 03 04 05; do
        provisioning_download "$HF_BASE/meituan-longcat/LongCat-Video/resolve/main/text_encoder/model-000${i}-of-00005.safetensors" "${WEIGHTS_DIR}/LongCat-Video/text_encoder" &
    done
    wait
    provisioning_download "$HF_BASE/meituan-longcat/LongCat-Video/resolve/main/text_encoder/model.safetensors.index.json" "${WEIGHTS_DIR}/LongCat-Video/text_encoder"
    provisioning_download "$HF_BASE/meituan-longcat/LongCat-Video/resolve/main/text_encoder/config.json" "${WEIGHTS_DIR}/LongCat-Video/text_encoder"

    echo "[$(date)] Tokenizer (21 MB)..."
    provisioning_download "$HF_BASE/meituan-longcat/LongCat-Video/resolve/main/tokenizer/spiece.model" "${WEIGHTS_DIR}/LongCat-Video/tokenizer"
    provisioning_download "$HF_BASE/meituan-longcat/LongCat-Video/resolve/main/tokenizer/tokenizer_config.json" "${WEIGHTS_DIR}/LongCat-Video/tokenizer"

    # ===== Avatar-1.5 =====
    echo "[$(date)] INT8 DiT (4 shards, 15.9 GB)..."
    for i in 01 02 03 04; do
        provisioning_download "$HF_BASE/meituan-longcat/LongCat-Video-Avatar-1.5/resolve/main/base_model_int8/quantized_model-000${i}-of-00004.safetensors" "${WEIGHTS_DIR}/LongCat-Video-Avatar-1.5/base_model_int8" &
    done
    wait
    provisioning_download "$HF_BASE/meituan-longcat/LongCat-Video-Avatar-1.5/resolve/main/base_model_int8/quantized_model.safetensors.index.json" "${WEIGHTS_DIR}/LongCat-Video-Avatar-1.5/base_model_int8"
    provisioning_download "$HF_BASE/meituan-longcat/LongCat-Video-Avatar-1.5/resolve/main/base_model_int8/config.json" "${WEIGHTS_DIR}/LongCat-Video-Avatar-1.5/base_model_int8"

    echo "[$(date)] DMD LoRA (2.5 GB)..."
    provisioning_download "$HF_BASE/meituan-longcat/LongCat-Video-Avatar-1.5/resolve/main/lora/dmd_lora.safetensors" "${WEIGHTS_DIR}/LongCat-Video-Avatar-1.5/lora"

    echo "[$(date)] Scheduler..."
    provisioning_download "$HF_BASE/meituan-longcat/LongCat-Video-Avatar-1.5/resolve/main/scheduler/scheduler_config.json" "${WEIGHTS_DIR}/LongCat-Video-Avatar-1.5/scheduler"

    echo "[$(date)] Vocal separator (67 MB)..."
    provisioning_download "$HF_BASE/meituan-longcat/LongCat-Video-Avatar-1.5/resolve/main/vocal_separator/Kim_Vocal_2.onnx" "${WEIGHTS_DIR}/LongCat-Video-Avatar-1.5/vocal_separator"

    echo "[$(date)] Whisper-large-v3 (~3.1 GB)..."
    provisioning_download "$HF_BASE/meituan-longcat/LongCat-Video-Avatar-1.5/resolve/main/whisper-large-v3/model.safetensors" "${WEIGHTS_DIR}/LongCat-Video-Avatar-1.5/whisper-large-v3"
    for f in added_tokens.json config.json generation_config.json merges.txt normalizer.json preprocessor_config.json special_tokens_map.json tokenizer.json tokenizer_config.json; do
        provisioning_download "$HF_BASE/meituan-longcat/LongCat-Video-Avatar-1.5/resolve/main/whisper-large-v3/$f" "${WEIGHTS_DIR}/LongCat-Video-Avatar-1.5/whisper-large-v3" &
    done
    wait

    echo "[$(date)] Pipeline configs..."
    provisioning_download "$HF_BASE/meituan-longcat/LongCat-Video-Avatar-1.5/resolve/main/config.json" "${WEIGHTS_DIR}/LongCat-Video-Avatar-1.5"
    provisioning_download "$HF_BASE/meituan-longcat/LongCat-Video-Avatar-1.5/resolve/main/model_index.json" "${WEIGHTS_DIR}/LongCat-Video-Avatar-1.5"

    echo "[$(date)] Model download complete."
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
