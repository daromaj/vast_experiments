#!/bin/bash

# ==============================================================================
# Vast.ai Provisioning Script for Wan 2.1 & InfiniteTalk (RTX 5090/Blackwell)
# ==============================================================================

# Logging setup
LOG_FILE="/workspace/provisioning.log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[$(date)] Starting Provisioning Script..."

# 1. ENVIRONMENT VARIABLES & PATHS
WORKSPACE="/workspace"
COMFY_DIR="$WORKSPACE/ComfyUI"
MODELS_DIR="$COMFY_DIR/models"
CUSTOM_NODES="$COMFY_DIR/custom_nodes"

# 2. INSTALL SYSTEM DEPENDENCIES
echo "[$(date)] Installing system dependencies..."
apt-get update
apt-get install -y git wget python3-dev python3-pip libgl1-mesa-glx libglib2.0-0 build-essential

# 3. BLACKWELL PYTORCH UPGRADE (CRITICAL)
# Remove incompatible versions provided by the base image
echo "[$(date)] Upgrading PyTorch for CUDA 12.8/Blackwell..."
pip uninstall -y torch torchvision torchaudio xformers
# Install Nightly Build for SM_120 support
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# 4. INSTALL COMFYUI (PERSISTENT INSTALL)
if; then
    echo "[$(date)] Cloning ComfyUI to persistent storage..."
    cd $WORKSPACE
    git clone https://github.com/comfyanonymous/ComfyUI.git
else
    echo "[$(date)] ComfyUI already exists in workspace."
fi

cd $COMFY_DIR
pip install -r requirements.txt

# 5. INSTALL CUSTOM NODES

# 5.1 ComfyUI Manager
mkdir -p $CUSTOM_NODES
cd $CUSTOM_NODES
if [! -d "ComfyUI-Manager" ]; then
    git clone https://github.com/ltdrdata/ComfyUI-Manager.git
fi

# 5.2 WanVideoWrapper
if; then
    echo "[$(date)] Installing WanVideoWrapper..."
    git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git
    cd ComfyUI-WanVideoWrapper
    pip install -r requirements.txt
    
    # 5.3 COMPILE SAGEATTENTION (SOURCE BUILD)
    echo "[$(date)] Compiling SageAttention from source..."
    pip uninstall -y sageattention # Remove any wheel-based install
    
    cd.. # Back to custom_nodes
    git clone https://github.com/thu-ml/SageAttention.git
    cd SageAttention
    # Compilation flags for optimization
    export EXT_PARALLEL=4
    export NVCC_APPEND_FLAGS="--threads 8"
    python3 setup.py install
    cd.. # Back to custom_nodes
fi

# 6. MODEL DOWNLOADS (PERSISTENT & RESUMABLE)
echo "[$(date)] Downloading Models..."

# Create Directories
mkdir -p $MODELS_DIR/diffusion_models
mkdir -p $MODELS_DIR/text_encoders
mkdir -p $MODELS_DIR/vae
mkdir -p $MODELS_DIR/clip_vision

# Function for safe download
download_model() {
    url=$1
    dir=$2
    filename=$(basename "$url")
    if [! -f "$dir/$filename" ]; then
        echo "Downloading $filename..."
        wget -nc -P "$dir" "$url"
    else
        echo "$filename already exists. Skipping."
    fi
}

# 6.1 Wan 2.1 I2V (FP8)
download_model "https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/I2V/Wan2_1-I2V-14B-480P_fp8_e4m3fn_scaled_KJ.safetensors" "$MODELS_DIR/diffusion_models"

# 6.2 InfiniteTalk (FP16)
download_model "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/InfiniteTalk/Wan2_1-InfiniTetalk-Single_fp16.safetensors" "$MODELS_DIR/diffusion_models"

# 6.3 VAE
download_model "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors" "$MODELS_DIR/vae"

# 6.4 Text Encoder (UMT5)
download_model "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5_xxl_fp8_e4m3fn_scaled.safetensors" "$MODELS_DIR/text_encoders"

# 6.5 Clip Vision
download_model "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors" "$MODELS_DIR/clip_vision"

echo "[$(date)] Provisioning Complete. Ready to launch."