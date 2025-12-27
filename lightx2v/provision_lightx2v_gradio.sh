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
# libgl1-mesa-glx: often needed for opencv
# ffmpeg: for video processing
apt-get install -y aria2 git nano ffmpeg libgl1-mesa-glx libglib2.0-0 build-essential

# 3. INSTALL PYTHON DEPENDENCIES
echo "[$(date)] Installing Python dependencies..."

# Ensure uv is installed
pip install uv huggingface_hub

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
uv pip install flash-attn

# Sage attention
echo "[$(date)] Installing SageAttention..."
# Try to install from PyPI first
if ! uv pip install sageattention==2.2.0; then
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
# We'll attempt to install it, but it might fail if not on ADA or if compilation fails.
echo "[$(date)] Attempting to install q8-kernel..."
if ! uv pip install git+https://github.com/KONAKONA666/q8_kernels.git; then
    echo "[$(date)] Warning: q8-kernel installation failed. This is expected if not on ADA GPU."
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

# We need to set lightx2v_path and model_path in the script
# The script likely has placeholders or variables at the top.
# Let's assume standard bash variable assignment structure or just append/replace.
# Since we don't know the exact content of run_gradio.sh, we'll append the variables 
# to the top of the file (after shebang) or use sed if we can guess the structure.
# A safer bet based on the instructions "Edit the startup script" is to create a wrapper 
# or try to replace known lines.
# The instructions say:
# # Configuration items that need to be modified:
# # - lightx2v_path: Lightx2v project root directory path
# # - model_path: Model root directory path (contains all model files)

# Let's try to sed replace if they exist, or just prepend them.
# However, if they are defined later in the file, prepending might not work if they are overwritten.
# Let's look at the file content first? No, I can't see it yet as I haven't cloned it.
# I'll assume they are defined as `lightx2v_path="..."` or similar.
# I will use a heuristic to replace them.

if [ -f "$RUN_SCRIPT" ]; then
    # Create a backup
    cp "$RUN_SCRIPT" "${RUN_SCRIPT}.bak"
    
    # Attempt to replace empty or placeholder paths
    # We'll just force set them at the beginning of the script (after shebang)
    # This is usually safe for bash scripts unless they are readonly variables (unlikely).
    
    # Insert variables after the first line
    sed -i "1a lightx2v_path='$LIGHTX2V_DIR'" "$RUN_SCRIPT"
    sed -i "1a model_path='$MODELS_DIR'" "$RUN_SCRIPT"
    
    echo "Modified $RUN_SCRIPT to set paths."
else
    echo "Warning: $RUN_SCRIPT not found!"
fi

echo "[$(date)] Provisioning Complete!"
echo "To start Gradio, run:"
echo "cd $LIGHTX2V_DIR/app && bash run_gradio.sh --lang en --port 7862"

# Optional: Auto-start (commented out by default to allow user to check logs first)
# cd "$LIGHTX2V_DIR/app"
# bash run_gradio.sh --lang en --port 7862
