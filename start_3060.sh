#!/bin/bash

echo "Starting Provisioning for RTX 3090..."
export DEBIAN_FRONTEND=noninteractive
apt-get update && apt-get install -y ffmpeg git aria2 nano libsox-fmt-all

# 1. Activate Environment
if command -v python3 >/dev/null 2>&1 && [ -f /venv/main/bin/activate ]; then
    . /venv/main/bin/activate
else
    echo "Virtual environment not found or not available. Using system Python."
fi

# 2. Install SageAttention (Critical for 3090 speed)
# As of Dec 2025, this installs the stable version compatible with Ampere cards
if ! pip show sageattention > /dev/null 2>&1; then
    pip install sageattention --no-build-isolation --break-system-packages 2>/dev/null || echo "Failed to install SageAttention via pip."
else
    echo "SageAttention is already installed, skipping installation."
fi

# Navigate to ComfyUI custom nodes directory
# mkdir -p /workspace/ComfyUI/custom_nodes
cd /workspace/ComfyUI/custom_nodes

# Install from Git repositories

# Kijai's Wrapper (Handles Wan 2.1, InfiniteTalk, and GGUF loading)
if [ ! -d "ComfyUI-WanVideoWrapper" ]; then
    echo "Cloning WanVideoWrapper..."
    git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git
    cd ComfyUI-WanVideoWrapper
    pip install -r requirements.txt --break-system-packages 2>/dev/null || echo "Failed to install requirements for WanVideoWrapper, trying system install..." && pip install -r requirements.txt
    cd ..
fi

# ComfyUI Manager
if [ ! -d "ComfyUI-Manager" ]; then
    git clone https://github.com/ltdrdata/ComfyUI-Manager.git
fi

# VideoHelperSuite (For video saving)
if [ ! -d "ComfyUI-VideoHelperSuite" ]; then
    echo "Cloning VideoHelperSuite..."
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
    cd ComfyUI-VideoHelperSuite
    pip install -r requirements.txt --break-system-packages 2>/dev/null || echo "Failed to install requirements for VideoHelperSuite, trying system install..." && pip install -r requirements.txt
    cd ..
fi

# 4. Model Acquisition (GGUF Optimized for 24GB VRAM)
MODEL_ROOT="/workspace/ComfyUI/models"

# 4.1 Wan 2.1 Diffusion Model (GGUF Q4_K_M - ~9GB)
# We use City96/Kijai's verified GGUF quant
mkdir -p "$MODEL_ROOT/diffusion_models"
echo "Downloading Wan 2.1 14B Q4_K_M GGUF..."
aria2c -x 16 -s 16 -k 1M -c -d "$MODEL_ROOT/diffusion_models" -o "wan2.1_i2v_14b_q4km.gguf" "https://huggingface.co/city96/Wan2.1-I2V-14B-480P-gguf/resolve/main/wan2.1-i2v-14b-480p-Q4_K_M.gguf"
echo "Downloading Wan 2.1 14B Q4_K_S GGUF..."
aria2c -x 16 -s 16 -k 1M -c -d "$MODEL_ROOT/diffusion_models" -o "wan2.1_i2v_14b_q4ks.gguf" "https://huggingface.co/city96/Wan2.1-I2V-14B-480P-gguf/resolve/main/wan2.1-i2v-14b-480p-Q4_K_S.gguf"

# 4.2 InfiniteTalk Adapter (GGUF Q4_K_M - ~1.4GB)
# This small adapter prevents VRAM spillover
echo "Downloading InfiniteTalk GGUF Adapter..."
aria2c -x 16 -s 16 -k 1M -c -d "$MODEL_ROOT/diffusion_models" -o "Wan2_1-InfiniteTalk-Single_Q4_K_M.gguf" "https://huggingface.co/Kijai/WanVideo_comfy_GGUF/resolve/main/InfiniteTalk/Wan2_1-InfiniteTalk_Single_Q4_K_M.gguf"

# 4.3 Text Encoder (UMT5-XXL FP8)
mkdir -p "$MODEL_ROOT/text_encoders"
# echo "Downloading UMT5 Text Encoder..."
# aria2c -x 16 -s 16 -k 1M -c -d "$MODEL_ROOT/text_encoders" -o "umt5_xxl_fp8_e4m3fn_scaled.safetensors" "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
echo "Downloading UMT5-XXL Encoder BF16..."
aria2c -x 16 -s 16 -k 1M -c -d "$MODEL_ROOT/text_encoders" -o "umt5-xxl-enc-bf16.safetensors" "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-bf16.safetensors"

# 4.4 VAE & Clip Vision (Standard)
mkdir -p "$MODEL_ROOT/vae"
mkdir -p "$MODEL_ROOT/clip_vision"
echo "Downloading VAE & Clip Vision..."
aria2c -x 16 -s 16 -k 1M -c -d "$MODEL_ROOT/vae" -o "wan_2.1_vae.safetensors" "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors"
aria2c -x 16 -s 16 -k 1M -c -d "$MODEL_ROOT/clip_vision" -o "clip_vision_h.safetensors" "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors"

# 4.5 Lightx2v Model (Distilled RANK64 BF16 - ~3.5GB)
echo "Downloading Lightx2v I2V Model..."
aria2c -x 16 -s 16 -k 1M -c -d "$MODEL_ROOT/loras" -o "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors" "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors?download=true"

# 4.6 InfiniteTalk Adapter (FP8 E4M3FN Scaled KJ - ~1.4GB)
echo "Downloading InfiniteTalk FP8 Adapter..."
aria2c -x 16 -s 16 -k 1M -c -d "$MODEL_ROOT/diffusion_models" -o "Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors" "https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/InfiniteTalk/Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors"

# 4.7 Wan 2.1 I2V FP8 Model (E4M3FN - ~14GB)
echo "Downloading Wan 2.1 I2V FP8 Model..."
aria2c -x 16 -s 16 -k 1M -c -d "$MODEL_ROOT/diffusion_models" -o "Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors" "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors"

# 4.8 MelBandRoFormer Model (FP16)
echo "Downloading MelBandRoFormer Model..."
aria2c -x 16 -s 16 -k 1M -c -d "$MODEL_ROOT/diffusion_models" -o "MelBandRoformer_fp16.safetensors" "https://huggingface.co/Kijai/MelBandRoFormer_comfy/resolve/6251b3a2bd544aaa31400138e55abda4722735cc/MelBandRoformer_fp16.safetensors"

echo "Provisioning for RTX 3090 Complete."
