#!/bin/bash

# ==============================================================================
# Vast.ai Provisioning Script for LightX2V with Qwen Models
# Target Image: vastai/pytorch:cuda-12.9.1-auto (or compatible PyTorch/CUDA image)
# ==============================================================================

# Logging setup
LOG_FILE="/workspace/provisioning.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[$(date)] Starting LightX2V Provisioning..."

# 1. ENVIRONMENT VARIABLES & PATHS
WORKSPACE="/workspace"
LIGHTX2V_DIR="$WORKSPACE/LightX2V"
ENV_ACTIVATE="/venv/main/bin/activate"

# Activate environment if it exists (Vast.ai standard)
if [ -f "$ENV_ACTIVATE" ]; then
    source "$ENV_ACTIVATE"
fi

# 2. INSTALL SYSTEM DEPENDENCIES
echo "[$(date)] Installing system dependencies..."
apt-get update
# aria2: for fast downloads
# libgl1-mesa-glx: often needed for opencv
# ffmpeg: for video processing
apt-get install -y aria2 git nano ffmpeg libgl1-mesa-glx libglib2.0-0 build-essential

# 3. INSTALL PYTHON DEPENDENCIES
echo "[$(date)] Installing Python dependencies..."

# Ensure uv is installed
pip install uv

# Clone LightX2V if not present
if [ ! -d "$LIGHTX2V_DIR" ]; then
    echo "[$(date)] Cloning LightX2V..."
    git clone https://github.com/ModelTC/LightX2V.git "$LIGHTX2V_DIR"
else
    echo "[$(date)] LightX2V already exists. Pulling latest..."
    cd "$LIGHTX2V_DIR" && git pull
fi

cd "$LIGHTX2V_DIR"

# Install LightX2V
echo "[$(date)] Installing LightX2V package..."
uv pip install -v .


# 3.5. INSTALL SAGEATTENTION
echo "[$(date)] Installing SageAttention..."
# User requested specific version without build isolation
pip install sageattention==2.2.0 --no-build-isolation


# 4. MODEL DOWNLOADS
echo "[$(date)] Downloading Models..."

MODELS_DIR="$LIGHTX2V_DIR/models"
mkdir -p "$MODELS_DIR"

# Helper function for aria2c download
download_model() {
    url=$1
    dir=$2
    filename=$(basename "$url")
    
    echo "Downloading $filename..."
    aria2c -x 16 -s 16 -k 1M -c --summary-interval=10 --console-log-level=notice \
        --allow-overwrite=true --auto-file-renaming=false \
        -d "$dir" "$url"
}

# 4.1 Qwen-Image
# https://huggingface.co/Qwen/Qwen-Image
# Warning: These models can be very large.
# We download to a 'Qwen' subdirectory for organization if LightX2V supports it, 
# otherwise we might need to adjust paths.
# Based on usage, usually you point to the model path.

QWEN_MODELS_DIR="$MODELS_DIR/Qwen"
mkdir -p "$QWEN_MODELS_DIR"

# Qwen-Image (Main Model)
# We need to download the full repo or specific files? 
# Usually for HF models, it's best to clone or download all files.
# Using python script to download snapshot is deeper but safer for folder structures.
echo "[$(date)] Downloading Qwen-Image via huggingface_hub..."
export HF_HUB_ENABLE_HF_TRANSFER=1
pip install huggingface_hub[cli] hf_transfer

huggingface-cli download --resume-download Qwen/Qwen-Image --local-dir "$QWEN_MODELS_DIR/Qwen-Image" --exclude "*.bin"  # Prefer safetensors if available, otherwise remove exclude

# 4.2 Qwen-Image-Edit-2511
echo "[$(date)] Downloading Qwen-Image-Edit-2511..."
huggingface-cli download --resume-download Qwen/Qwen-Image-Edit-2511 --local-dir "$QWEN_MODELS_DIR/Qwen-Image-Edit-2511"


# 4.3 (Optional) Qwen-Image-Edit-2511-Lightning (FP8/Distilled)
echo "[$(date)] Downloading Qwen-Image-Edit-2511-Lightning (Accelerated)..."
huggingface-cli download --resume-download lightx2v/Qwen-Image-Edit-2511-Lightning --local-dir "$QWEN_MODELS_DIR/Qwen-Image-Edit-2511-Lightning"


# 5. DOWNLOAD EXAMPLE SCRIPTS
echo "[$(date)] Downloading example scripts..."
SCRIPTS_DIR="$LIGHTX2V_DIR/examples/qwen_custom"
mkdir -p "$SCRIPTS_DIR"

# Raw URLs for the scripts
GEN_SCRIPT_URL="https://raw.githubusercontent.com/daromaj/vast_experiments/master/lightx2v/generate_image.py"
EDIT_SCRIPT_URL="https://raw.githubusercontent.com/daromaj/vast_experiments/master/lightx2v/edit_image.py"

echo "Downloading generate_image.py..."
wget -O "$SCRIPTS_DIR/generate_image.py" "$GEN_SCRIPT_URL"

echo "Downloading edit_image.py..."
wget -O "$SCRIPTS_DIR/edit_image.py" "$EDIT_SCRIPT_URL"

echo "[$(date)] Provisioning Complete!"
