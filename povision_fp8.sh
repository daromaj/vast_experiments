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
)

WORKFLOWS=(
    # "https://raw.githubusercontent.com/vast-ai/base-image/refs/heads/main/derivatives/pytorch/derivatives/comfyui/workflows/text_to_video_wan.json"
    "https://raw.githubusercontent.com/daromaj/vast_experiments/refs/heads/master/InfiniteTalk-I2V-FP8-Lip-Sync.json"
    "https://raw.githubusercontent.com/daromaj/vast_experiments/refs/heads/master/workflows/InfiniteTalk-I2V-FP8-Lip-Sync_5090_sage_new_prompts.json"
)

VAE_MODELS=(
    # "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors"
    "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors"
)

CLIP_VISION=(
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors"
)

LORAS=(
    "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"
)

TEXT_ENCODERS=(
    # "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
    "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-bf16.safetensors"
    "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-fp8_e4m3fn.safetensors"
)

DIFFUSION_MODELS=(
    # "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors"
    "https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/InfiniteTalk/Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors"
    "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors"
    "https://huggingface.co/Kijai/MelBandRoFormer_comfy/resolve/6251b3a2bd544aaa31400138e55abda4722735cc/MelBandRoformer_fp16.safetensors"

)

SAGEATTENTION_WHEELS=(
    "https://github.com/daromaj/vast_experiments/raw/master/python/sageattn3-1.0.0-cp312-cp312-linux_x86_64.whl"
    "https://github.com/daromaj/vast_experiments/raw/master/python/sageattention-2.2.0-cp312-cp312-linux_x86_64.whl"
    "https://github.com/daromaj/vast_experiments/raw/master/python/sageattention-2.2.0-cp312-cp312-linux_x86_64_4090.whl"
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
    # provisioning_install_sageattention  # Disabled: using source build instead
    # Start SageAttention build in background (CPU/GPU compilation) while downloads run (network I/O)
    # Output is prefixed with [SAGE] to distinguish from download progress
    { provisioning_install_sageattention_source 2>&1 | sed 's/^/[SAGE] /'; } &
    SAGE_BUILD_PID=$!

    workflows_dir="${COMFYUI_DIR}/user/default/workflows"
    mkdir -p "${workflows_dir}"
    provisioning_get_files "${workflows_dir}" "${WORKFLOWS[@]}"
    provisioning_get_files "${COMFYUI_DIR}/models/vae" "${VAE_MODELS[@]}"
    provisioning_get_files "${COMFYUI_DIR}/models/text_encoders" "${TEXT_ENCODERS[@]}"
    provisioning_get_files "${COMFYUI_DIR}/models/diffusion_models" "${DIFFUSION_MODELS[@]}"
    provisioning_get_files \
        "${COMFYUI_DIR}/models/loras" \
        "${LORAS[@]}"    

    provisioning_get_files "${COMFYUI_DIR}/models/clip_vision" "${CLIP_VISION[@]}"

    # Wait for SageAttention build to complete before finishing provisioning
    echo "Waiting for SageAttention build to complete..."
    wait $SAGE_BUILD_PID
    SAGE_EXIT_CODE=$?
    if [[ $SAGE_EXIT_CODE -ne 0 ]]; then
        echo "WARNING: SageAttention build failed with exit code $SAGE_EXIT_CODE - check logs above"
    else
        echo "SageAttention build completed successfully"
    fi

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
    # DEPRECATED: Wheel-based installation - kept for reference
    echo "Installing SageAttention from wheel files..."
    local wheel_dir="${WORKSPACE}/wheels"
    mkdir -p "$wheel_dir"

    # Download wheel files
    for url in "${SAGEATTENTION_WHEELS[@]}"; do
        provisioning_download "$url" "$wheel_dir"
    done

    # Install all downloaded wheels
    pip install --no-cache-dir "$wheel_dir"/*.whl
}

function provisioning_install_sageattention_source() {
    # Builds SageAttention from source with parallel compilation
    # Additionally builds sageattention3_blackwell if running on a 5090 GPU
    echo "Building SageAttention from source..."
    
    local sage_dir="${WORKSPACE}/SageAttention"
    
    # Clone the repository if not already present
    if [[ ! -d "$sage_dir" ]]; then
        echo "Cloning SageAttention repository..."
        git clone https://github.com/thu-ml/SageAttention.git "$sage_dir"
    else
        echo "SageAttention directory exists, pulling latest..."
        ( cd "$sage_dir" && git pull )
    fi
    
    # Set parallel compilation environment variables for faster builds
    export EXT_PARALLEL=4
    export NVCC_APPEND_FLAGS="--threads 8"
    export MAX_JOBS=32
    
    # Build and install main SageAttention package
    echo "Installing SageAttention (this may take a while)..."
    ( cd "$sage_dir" && python setup.py install )
    
    # Check if running on Blackwell (5090) GPU and build sageattention3 if so
    if nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -qi "5090"; then
        echo "Blackwell GPU (5090) detected - building sageattention3_blackwell..."
        local blackwell_dir="${sage_dir}/sageattention3_blackwell"
        if [[ -d "$blackwell_dir" ]]; then
            # doesn't seem to help with speed for now
            # ( cd "$blackwell_dir" && python setup.py install )
            echo "sageattention3_blackwell installed successfully"
        else
            echo "WARNING: sageattention3_blackwell directory not found at $blackwell_dir"
        fi
    else
        echo "Non-Blackwell GPU detected - skipping sageattention3_blackwell build"
    fi
    
    echo "SageAttention source build complete"
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
    local auth_header=""
    local filename=""

    # Extract filename from URL (remove query parameters and get last path segment)
    filename=$(basename "${url%%\?*}")

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
