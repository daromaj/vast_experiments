#!/bin/bash

source /venv/main/bin/activate
COMFYUI_DIR=${WORKSPACE}/ComfyUI

APT_PACKAGES=(aria2)
PIP_PACKAGES=(
)
NODES=(
    "https://github.com/kijai/ComfyUI-WanVideoWrapper.git"
    "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"
    "https://github.com/kijai/ComfyUI-MelBandRoFormer"
    "https://github.com/kijai/ComfyUI-KJNodes"
    "https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git"
    "https://github.com/cubiq/ComfyUI_essentials.git"
    "https://github.com/chrisgoringe/cg-use-everywhere.git"
    "https://github.com/AIFSH/ComfyUI-mxToolkit.git"
    "https://github.com/chibi-lamp/ComfyUI-Chibi-Nodes.git"
    "https://github.com/rgthree/rgthree-comfy.git"
    "https://github.com/crystian/ComfyUI-Crystools.git"
    "https://github.com/PGCRT/CRT-Nodes.git"
    # z-image-turbo
    "https://github.com/ltdrdata/ComfyUI-Impact-Pack.git"
    "https://github.com/yolain/ComfyUI-Easy-Use.git"
    "https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git"
    "https://github.com/ssitu/ComfyUI_UltimateSDUpscale.git"
    "https://github.com/sipherxyz/comfyui-art-venture.git"
    "https://github.com/ltdrdata/ComfyUI-Impact-Subpack.git"
    "https://github.com/gseth/ControlAltAI-Nodes.git"
    "https://github.com/omar92/ComfyUI-QualityOfLifeSuit_Omar92.git"
    "https://github.com/vrgamegirl19/comfyui-vrgamedevgirl.git"
    "https://github.com/giriss/comfy-image-saver.git"
)

WORKFLOWS=(
    # "https://raw.githubusercontent.com/vast-ai/base-image/refs/heads/main/derivatives/pytorch/derivatives/comfyui/workflows/text_to_video_wan.json"
    # "https://raw.githubusercontent.com/daromaj/vast_experiments/refs/heads/master/InfiniteTalk-I2V-FP8-Lip-Sync.json"
    # "https://raw.githubusercontent.com/daromaj/vast_experiments/refs/heads/master/workflows/InfiniteTalk-I2V-FP8-Lip-Sync_5090_sage_new_prompts.json"
    "https://raw.githubusercontent.com/daromaj/vast_experiments/refs/heads/master/workflows/WAN%202.2%20I2V.json"
    "https://raw.githubusercontent.com/daromaj/vast_experiments/refs/heads/master/workflows/WAN%202.2%20T2V.json"
    "https://raw.githubusercontent.com/daromaj/vast_experiments/refs/heads/master/workflows/WAN%202.2%20I2V%20StartEnd%20Frames.json"
)

VAE_MODELS=(
    # "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors"
    "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors"
    "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_2_VAE_bf16.safetensors"
    "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors"
)

CLIP_VISION=(
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors"
)

LORAS=(
    # "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"
)

TEXT_ENCODERS=(
    # "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
    "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-bf16.safetensors"
    "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-fp8_e4m3fn.safetensors"    
    "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors"
)

DIFFUSION_MODELS=(
    # "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors"
    # "https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/InfiniteTalk/Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors"
    "https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/S2V/Wan2_2-S2V-14B_fp8_e4m3fn_scaled_KJ.safetensors"
    "https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/I2V/Wan2_2-I2V-A14B-HIGH_fp8_e4m3fn_scaled_KJ.safetensors"
    "https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/I2V/Wan2_2-I2V-A14B-LOW_fp8_e4m3fn_scaled_KJ.safetensors"
    # "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors"
    "https://huggingface.co/Kijai/MelBandRoFormer_comfy/resolve/6251b3a2bd544aaa31400138e55abda4722735cc/MelBandRoformer_fp16.safetensors"
    "https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/T2V/Wan2_2-T2V-A14B_HIGH_fp8_e4m3fn_scaled_KJ.safetensors?download=true"
    "https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/T2V/Wan2_2-T2V-A14B-LOW_fp8_e4m3fn_scaled_KJ.safetensors?download=true"
    "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors"
)

function provisioning_start() {
    # Setup logging
    LOG_FILE="${WORKSPACE}/provisioning.log" 
    exec > >(tee -a "$LOG_FILE") 2>&1
    echo "[$(date)] Starting provisioning..."

    # Pre-flight check for aria2c 
    if ! command -v aria2c &> /dev/null; then
        echo "NOTICE: aria2c not found - will be installed via APT_PACKAGES"
    fi

    provisioning_print_header
    provisioning_get_apt_packages
    provisioning_get_nodes
    provisioning_get_pip_packages
    provisioning_install_sageattention

    workflows_dir="${COMFYUI_DIR}/user/default/workflows"
    mkdir -p "${workflows_dir}"
    provisioning_get_files "${workflows_dir}" "${WORKFLOWS[@]}"
    provisioning_get_files "${COMFYUI_DIR}/models/vae" "${VAE_MODELS[@]}"
    provisioning_get_files "${COMFYUI_DIR}/models/text_encoders" "${TEXT_ENCODERS[@]}"
    provisioning_get_files "${COMFYUI_DIR}/models/diffusion_models" "${DIFFUSION_MODELS[@]}"
    provisioning_get_files \
        "${COMFYUI_DIR}/models/loras" \
        "${LORAS[@]}"    

    provisioning_download "https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V2.0/high_noise_model.safetensors?download=true" \
        "${COMFYUI_DIR}/models/loras" \
        "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V2.0_high_noise.safetensors"

    provisioning_download "https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V2.0/low_noise_model.safetensors?download=true" \
        "${COMFYUI_DIR}/models/loras" \
        "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V2.0_low_noise.safetensors"

    provisioning_download "https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors?download=true" \
        "${COMFYUI_DIR}/models/loras" \
        "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1_high_noise.safetensors"

    provisioning_download "https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors?download=true" \
        "${COMFYUI_DIR}/models/loras" \
        "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1_low_noise.safetensors"

    provisioning_get_files "${COMFYUI_DIR}/models/clip_vision" "${CLIP_VISION[@]}"
    provisioning_download "https://huggingface.co/moxeeeem/wav2vec2-finetuned-pronunciation-correction/resolve/main/model.safetensors" \
        "${COMFYUI_DIR}/models/audio_encoders" \
        "wav2vec2-finetuned-pronunciation-correction.safetensors"
    # z-image-turbo
    mkdir -p "${COMFYUI_DIR}/models/ultralytics/bbox"
    provisioning_download "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt?download=true" \
        "${COMFYUI_DIR}/models/ultralytics/bbox" \
        "face_yolov8m.pt"

    provisioning_download "https://huggingface.co/Tenofas/ComfyUI/resolve/d79945fb5c16e8aef8a1eb3ba1788d72152c6d96/ultralytics/bbox/Eyeful_v2-Paired.pt?download=true" \
        "${COMFYUI_DIR}/models/ultralytics/bbox" \
        "Eyeful_v2-Paired.pt"

    provisioning_download "https://huggingface.co/xingren23/comfyflow-models/resolve/main/ultralytics/bbox/hand_yolov8s.pt?download=true" \
        "${COMFYUI_DIR}/models/ultralytics/bbox" \
        "hand_yolov8s.pt"

    mkdir -p "${COMFYUI_DIR}/models/SEEDVR2"
    provisioning_download "https://huggingface.co/cmeka/SeedVR2-GGUF/resolve/main/seedvr2_ema_7b-Q4_K_M.gguf?download=true" \
        "${COMFYUI_DIR}/models/SEEDVR2" \
        "seedvr2_ema_7b-Q4_K_M.gguf"

    provisioning_print_end
}

function provisioning_get_apt_packages() {
    if [[ -n $APT_PACKAGES ]]; then
        # Use APT_INSTALL if defined, otherwise fallback to apt-get
        local apt_cmd="${APT_INSTALL:-apt-get install -y}"
        sudo $apt_cmd ${APT_PACKAGES[@]}
    fi
}

function provisioning_get_pip_packages() {
    if [[ -n $PIP_PACKAGES ]]; then
        pip install --no-cache-dir ${PIP_PACKAGES[@]}
    fi
}

function provisioning_install_sageattention() {
    echo "Installing SageAttention..."
    local repo="https://github.com/thu-ml/SageAttention.git"
    local path="${WORKSPACE}/SageAttention"
    # pip install /tmp/sageattention-*.whl

    if [[ ! -d $path ]]; then
        echo "Cloning SageAttention..."
        git clone "${repo}" "${path}"
    else
        echo "SageAttention directory already exists, skipping clone."
    fi
    if [[ -d $path ]]; then
        echo "Installing SageAttention..."
        ( cd "$path" && export EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=32 && cd SageAttention/sageattention3_blackwell && python setup.py install )
    fi
}

function provisioning_get_nodes() {
    for repo in "${NODES[@]}"; do
        dir="${repo##*/}"
        path="${COMFYUI_DIR}/custom_nodes/${dir}"
        requirements="${path}/requirements.txt"
        if [[ -d $path ]]; then
            if [[ ${AUTO_UPDATE,,} != "false" ]]; then
                echo "Updating node: ${repo}"
                ( cd "$path" && git pull )
                [[ -e $requirements ]] && pip install --no-cache-dir -r "$requirements"
            fi
        else
            echo "Downloading node: ${repo}"
            git clone "${repo}" "${path}" --recursive
            [[ -e $requirements ]] && pip install --no-cache-dir -r "$requirements"
        fi
    done
}

function provisioning_get_files() {
    [[ -z $2 ]] && return 1
    dir="$1"
    mkdir -p "$dir"
    shift
    arr=("$@")
    echo "Downloading ${#arr[@]} file(s) to $dir..."
    for url in "${arr[@]}"; do
        echo "Downloading: $url"
        provisioning_download "$url" "$dir"
        echo
    done
}

function provisioning_print_header() {
    echo -e "\\n##############################################"
    echo -e "#          Provisioning container            #"
    echo -e "#         This will take some time           #"
    echo -e "# Your container will be ready on completion #"
    echo -e "##############################################\\n"
}

function provisioning_print_end() {
    echo -e "\\nProvisioning complete: Application will start now\\n"
}

function provisioning_download() {
    local url="$1"
    local dir="$2"
    local filename="$3"
    local auth_header=""

    if [[ -z $filename ]]; then
        filename=$(basename "${url%%\?*}")
    fi

    # Detect HuggingFace URLs and add auth if token exists
    if [[ -n $HF_TOKEN && $url =~ ^https://([a-zA-Z0-9_-]+\\.)?huggingface\\.co(/|$|\\?) ]]; then
        auth_header="--header=Authorization: Bearer $HF_TOKEN"
    fi

    # Use aria2c with optimal settings (16 parallel connections, auto-resume)
    # -o: Explicit output filename to avoid hash-based names
    # --summary-interval=10: Show progress every 10 seconds (default is 60)
    # --console-log-level=notice: Show download progress and errors
    # --allow-overwrite=true: Allow overwriting existing files
    # --auto-file-renaming=false: Don't rename files automatically
    if [[ -n $auth_header ]]; then
        aria2c -x 16 -s 16 -k 1M -c --summary-interval=10 --console-log-level=notice \
            --allow-overwrite=true --auto-file-renaming=false \
            -o "$filename" $auth_header -d "$dir" "$url"
    else
        aria2c -x 16 -s 16 -k 1M -c --summary-interval=10 --console-log-level=notice \
            --allow-overwrite=true --auto-file-renaming=false \
            -o "$filename" -d "$dir" "$url"
    fi

    # Note: No explicit error handling - continue on failures, check logs later
}

if [[ ! -f /.noprovisioning ]]; then
    provisioning_start
fi
