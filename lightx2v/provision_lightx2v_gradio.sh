#!/bin/bash

# ==============================================================================
# Vast.ai Provisioning Script for LightX2V with Gradio
# Target Image: vastai/pytorch:cuda-12.9.1-auto (or compatible PyTorch/CUDA image)
# ==============================================================================

# Logging setup
LOG_FILE="/workspace/provisioning.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[$(date)] Starting LightX2V Gradio Provisioning..."

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
# libgl1 libglx-mesa0: for opencv (libgl1-mesa-glx is obsolete in Ubuntu 24.04)
# ffmpeg: for video processing
apt-get install -y aria2 git nano ffmpeg libgl1 libglx-mesa0 libglib2.0-0 build-essential

# 3. INSTALL PYTHON DEPENDENCIES
echo "[$(date)] Installing Python dependencies..."

# Ensure uv is installed
pip install uv huggingface_hub[cli] hf_transfer modelscope

# Enable hf_transfer for faster downloads
export HF_HUB_ENABLE_HF_TRANSFER=1

# Install server dependencies (if needed, though Gradio is the main one here)
# The repo requirements likely handle most, but we'll install the optimized kernels explicitly as requested.

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

# Install Recommended Optimization Libraries
echo "[$(date)] Installing Optimization Libraries..."
# Flash attention (usually pre-installed in some docker images, but good to ensure)
uv pip install flash-attn --no-build-isolation

# Sage attention
echo "[$(date)] Installing SageAttention..."
# Try to install from PyPI first
if ! uv pip install sageattention==2.2.0 --no-build-isolation; then
    echo "[$(date)] pip install failed, building SageAttention from source..."
    cd "$WORKSPACE"
    git clone https://github.com/thu-ml/SageAttention.git
    cd SageAttention
    # Optimization flags
    export EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=32
    python setup.py install
    cd "$LIGHTX2V_DIR"
fi

# vllm-kernel & sgl-kernel
uv pip install vllm sgl-kernel

# q8-kernel (Only for ADA architecture GPUs)
# Ada Lovelace has compute capability 8.9
echo "[$(date)] Checking GPU architecture..."
if command -v nvidia-smi &> /dev/null; then
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
    echo "Detected Compute Capability: $COMPUTE_CAP"
    
    # Check if compute capability is 8.9 (Ada)
    if [[ "$COMPUTE_CAP" == "8.9" ]]; then
        echo "[$(date)] Ada architecture detected. Installing q8-kernel..."
        if ! uv pip install git+https://github.com/KONAKONA666/q8_kernels.git; then
             echo "[$(date)] Warning: q8-kernel installation failed."
        fi
    else
        echo "[$(date)] Compute Capability $COMPUTE_CAP is not 8.9 (Ada). Skipping q8-kernel."
    fi
else
    echo "[$(date)] nvidia-smi not found. Skipping GPU architecture check and q8-kernel installation."
fi

# 4. MODEL DOWNLOADS
echo "[$(date)] Downloading Models..."

MODELS_DIR="$LIGHTX2V_DIR/models"
mkdir -p "$MODELS_DIR"

# 4.1 Wan2.1 Model (Distilled)
# Downloading Wan2.1 720p model
echo "[$(date)] Downloading Wan2.1 720p model..."
hf download lightx2v/Wan2.1-Distill-Models wan2.1_i2v_720p_lightx2v_4step.safetensors --local-dir "$MODELS_DIR"
hf download lightx2v/Wan2.1-Distill-Models config.json --local-dir "$MODELS_DIR"

# 4.2 Encoders
# "Text and Image Encoders can be downloaded from Encoders"
echo "[$(date)] Downloading Encoders..."
hf download lightx2v/Encoders --local-dir "$MODELS_DIR"

# 4.3 VAE
# "VAE can be downloaded from Autoencoders"
echo "[$(date)] Downloading VAE..."
hf download lightx2v/Autoencoders --local-dir "$MODELS_DIR"


# 5. CONFIGURE STARTUP SCRIPT
echo "[$(date)] Configuring startup script..."
RUN_SCRIPT="$LIGHTX2V_DIR/app/run_gradio.sh"

if [ -f "$RUN_SCRIPT" ]; then
    # Create a backup
    cp "$RUN_SCRIPT" "${RUN_SCRIPT}.bak"
    
    # Use sed to replace existing variable definitions
    # Matches lines starting with optional whitespace/comment, then variable name, then =
    # We use | as delimiter to avoid issues with path slashes
    sed -i "s|^[#[:space:]]*lightx2v_path=.*|lightx2v_path='$LIGHTX2V_DIR'|" "$RUN_SCRIPT"
    sed -i "s|^[#[:space:]]*model_path=.*|model_path='$MODELS_DIR'|" "$RUN_SCRIPT"
    
    chmod +x "$RUN_SCRIPT"
    echo "Modified $RUN_SCRIPT to set paths."
    
    # Verify the change
    grep "lightx2v_path=" "$RUN_SCRIPT"
    grep "model_path=" "$RUN_SCRIPT"
else
    echo "Warning: $RUN_SCRIPT not found!"
fi

echo "[$(date)] Provisioning Complete!"
# 6. START SERVER
echo "[$(date)] Starting Gradio Server..."
cd "$LIGHTX2V_DIR/app"
# Start in background with nohup so it persists after script finishes
nohup bash run_gradio.sh --lang en --port 7862 > /workspace/gradio.log 2>&1 &
echo "[$(date)] Gradio server started in background. Logs at /workspace/gradio.log"
echo "Access at http://<instance-ip>:7862"
