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

# Install server dependencies
echo "[$(date)] Installing server dependencies..."
pip install fastapi uvicorn python-multipart

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
# SageAttention is disabled as it caused runtime errors (NoneType not callable)
# and we use standard torch_sdpa instead.
# echo "[$(date)] Installing SageAttention..."
# # Try to install from PyPI first
# if ! pip install sageattention==2.2.0 --no-build-isolation; then
#     echo "[$(date)] pip install failed, building SageAttention from source..."
#     cd "$WORKSPACE"
#     git clone https://github.com/thu-ml/SageAttention.git
#     cd SageAttention
#     export EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=32
#     python setup.py install
#     cd "$LIGHTX2V_DIR"
# fi


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
# We can likely skip the base Qwen-Image if we are using the Edit model for T2I
# echo "[$(date)] Downloading Qwen-Image via huggingface_hub..."
# huggingface-cli download --resume-download Qwen/Qwen-Image --local-dir "$QWEN_MODELS_DIR/Qwen-Image" --exclude "*.bin"

# 4.2 Qwen-Image-Edit-2511 (Complete Model)
# Download the full model including transformer weights (~55GB total)
# This includes:
#   - Text Encoder: ~16GB (4 safetensors files)
#   - Transformer: ~39GB (5 safetensors files)
#   - VAE: included in base model
# We use CPU offload to fit this in 32GB VRAM
echo "[$(date)] Downloading Qwen-Image-Edit-2511 (complete model with transformer)..."
hf download Qwen/Qwen-Image-Edit-2511 \
    --local-dir "$QWEN_MODELS_DIR/Qwen-Image-Edit-2511"

# Note: FP8 quantized checkpoint is NOT downloaded because:
# - sgl-kernel has initialization issues
# - Full-precision model works perfectly with CPU offload on 32GB GPU


# 5. DOWNLOAD EXAMPLE SCRIPT
echo "[$(date)] Downloading edit_image.py script..."
SCRIPTS_DIR="/workspace"

# Raw URL for the script
EDIT_SCRIPT_URL="https://raw.githubusercontent.com/daromaj/vast_experiments/master/lightx2v/edit_image.py"

echo "Downloading edit_image.py to /workspace..."
wget -O "$SCRIPTS_DIR/edit_image.py" "$EDIT_SCRIPT_URL"

# Raw URL for the server script
SERVER_SCRIPT_URL="https://raw.githubusercontent.com/daromaj/vast_experiments/master/lightx2v/server.py"

echo "Downloading server.py to /workspace..."
wget -O "$SCRIPTS_DIR/server.py" "$SERVER_SCRIPT_URL"

echo "Downloading example_edit_input.json..."
wget -O "$SCRIPTS_DIR/example_edit_input.json" "https://raw.githubusercontent.com/daromaj/vast_experiments/master/lightx2v/example_edit_input.json"

# Raw URL for the example server request script
EXAMPLE_SERVER_SCRIPT_URL="https://raw.githubusercontent.com/daromaj/vast_experiments/master/lightx2v/example_server_request.sh"

echo "Downloading example_server_request.sh..."
wget -O "$SCRIPTS_DIR/example_server_request.sh" "$EXAMPLE_SERVER_SCRIPT_URL"
chmod +x "$SCRIPTS_DIR/example_server_request.sh"

echo "[$(date)] Provisioning Complete!"

# 6. START SERVER
echo "[$(date)] Starting Image Edit Server..."
# Start server in background but keep logging to file
cd /workspace
nohup uvicorn server:app --host 0.0.0.0 --port 8000 > /workspace/server.log 2>&1 &
echo "[$(date)] Server started in background. Logs at /workspace/server.log"
