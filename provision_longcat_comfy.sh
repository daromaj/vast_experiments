#!/bin/bash
#
# Provisioning script for Vast.ai — LongCat-Video-Avatar-1.5 via ComfyUI + kijai WanVideoWrapper.
#
# This is the ComfyUI path (mirrors povision_fp8.sh / InfiniteTalk), NOT the dead
# raw-torchrun INT8 path in longcat_video/. Same stack you already run for InfiniteTalk:
# WanVideoWrapper + wav2vec/MultiTalk embeds + block swap + SageAttention.
#
# GPU targets: RTX 5090 (32GB) and RTX 4090 (24GB). fp8 DiT fits both.
#
# Total Download Size: ~29.4 GB
# Key Model Sizes:
# - LongCat-Avatar-single_fp8_e4m3fn_scaled_mixed_KJ.safetensors: ~16.9 GB  (DiT, fp8)
# - umt5-xxl-enc-bf16.safetensors:                                ~10.58 GB (text encoder, shared w/ InfiniteTalk)
# - LongCat-Avatar-15_dmd_distill_lora_rank128_bf16.safetensors:  ~1.26 GB  (8-step DMD distill LoRA)
# - MelBandRoformer_fp16.safetensors:                             ~435 MB   (vocal separator)
# - wav2vec2-chinese-base_fp16.safetensors:                       ~190 MB   (Wav2VecModelLoader, models/wav2vec2)
# - Wan2_1_VAE_bf16.safetensors:                                  ~242 MB
#
# After provisioning, in the ComfyUI UI:
#   1. Model loader: point at LongCat-Avatar-single_fp8_e4m3fn_scaled_mixed_KJ.safetensors
#      (the official example JSON defaults to the bf16 name; switch the dropdown to the fp8 file).
#   2. Block swap:  5090 → blocks_to_swap 0 (all resident). 4090 → 20 (raise to 25 if OOM, lower if headroom).
#   3. Steps:       start 8 (distill LoRA is DMD 8-step). Bump to 12 only if quality needs it.
#   4. Resolution:  832x480 for the 15-30 min / 60s budget. 720p blows past 30 min.
#
# Monitor Output: same [PROGRESS] / [PROG_DATA] JSON format as povision_fp8.sh.
#

source /venv/main/bin/activate
COMFYUI_DIR=${WORKSPACE}/ComfyUI
# fp8 DiT 16.9 + umt5 10.58 + lora 1.26 + melband 0.44 + vae 0.24 + wav2vec 0.19 ≈ 29.6 GB
TOTAL_BYTES_TO_DOWNLOAD=29590000000
MIN_SETUP_TIME=360  # Minimum 6 minutes for nodes + SageAttention build

APT_PACKAGES=(aria2 bc libcusparse-dev-12-9 libcublas-dev-12-9 libcusolver-dev-12-9 libcufft-dev-12-9 libcurand-dev-12-9)
PIP_PACKAGES=(
    # hf_transfer (Rust multi-threaded downloader) + the `hf` CLI for HuggingFace pulls.
    "huggingface_hub[hf_transfer]"
)
NODES=(
    "https://github.com/kijai/ComfyUI-WanVideoWrapper.git"
    "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"
    "https://github.com/kijai/ComfyUI-MelBandRoFormer"
    "https://github.com/kijai/ComfyUI-KJNodes"
)

WORKFLOWS=(
    # Official kijai LongCat-Avatar audio+image → video example (832x480, 93f window, distill LoRA, sageattn).
    "https://raw.githubusercontent.com/kijai/ComfyUI-WanVideoWrapper/main/example_workflows/LongCatAvatar_audio_image_to_video_example_01.json"
)

VAE_MODELS=(
    "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors"
)

LORAS=(
    # 8-step DMD distillation LoRA for LongCat-Avatar-1.5
    "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/LongCat/LongCat-Avatar-15_dmd_distill_lora_rank128_bf16.safetensors"
)

TEXT_ENCODERS=(
    # bf16 encoder — same file the InfiniteTalk workflows already use.
    "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-bf16.safetensors"
)

WAV2VEC_MODELS=(
    # Single-file wav2vec for the current wrapper's Wav2VecModelLoader (reads ComfyUI/models/wav2vec2).
    # The loader carries its own wav2vec2_config.json, so only this .safetensors is needed.
    "https://huggingface.co/Kijai/wav2vec2_safetensors/resolve/main/wav2vec2-chinese-base_fp16.safetensors"
)

DIFFUSION_MODELS=(
    # fp8 DiT — fits both 5090 (resident) and 4090 (with block swap). Primary path.
    "https://huggingface.co/Kijai/LongCat-Video_comfy/resolve/main/Avatar/LongCat-Avatar-single_fp8_e4m3fn_scaled_mixed_KJ.safetensors"
    # bf16 reference DiT (31.7 GB) — only for 5090 max-quality runs, does NOT fit a 4090. Uncomment if wanted.
    # "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/LongCat/LongCat-Avatar-15_bf16.safetensors"
    # Vocal separator (reused from InfiniteTalk stack). Bypass in the UI if input audio is already clean voice.
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
    local provisioning_start_time=$(date +%s)
    echo "[$(date)] Starting provisioning..."

    # Pre-flight check for aria2c
    if ! command -v aria2c &> /dev/null; then
        echo "NOTICE: aria2c not found - will be installed via APT_PACKAGES"
    fi

    provisioning_print_header
    provisioning_get_apt_packages
    provisioning_get_pip_packages



    # Start Download Monitoring in background
    provisioning_monitor_loop &
    MONITOR_PID=$!

    # Start Parallel Operations
    # 1. Build SageAttention (CPU intensive, no pip lock)
    # 2. Setup Nodes (Network/Disk intensive, locks pip)
    echo "Starting parallel setup: SageAttention Build + Node Installation..."

    { provisioning_build_sageattention 2>&1 | sed 's/^/[SAGE_BUILD] /'; } &
    SAGE_PID=$!

    { provisioning_get_nodes 2>&1 | sed 's/^/[NODES] /'; } &
    NODES_PID=$!



    workflows_dir="${COMFYUI_DIR}/user/default/workflows"
    mkdir -p "${workflows_dir}"
    provisioning_get_files "${workflows_dir}" "${WORKFLOWS[@]}"
    provisioning_get_files "${COMFYUI_DIR}/models/vae" "${VAE_MODELS[@]}"
    provisioning_get_files "${COMFYUI_DIR}/models/text_encoders" "${TEXT_ENCODERS[@]}"
    provisioning_get_files "${COMFYUI_DIR}/models/diffusion_models" "${DIFFUSION_MODELS[@]}"
    provisioning_get_files "${COMFYUI_DIR}/models/loras" "${LORAS[@]}"
    provisioning_get_files "${COMFYUI_DIR}/models/wav2vec2" "${WAV2VEC_MODELS[@]}"

    # Wait for both background processes
    echo "Waiting for background setup (SageAttention Build + Nodes) to complete..."
    wait $SAGE_PID
    local sage_status=$?

    wait $NODES_PID
    local nodes_status=$?

    if [[ $sage_status -eq 0 && $nodes_status -eq 0 ]]; then
        echo "Parallel setup complete. Installing SageAttention..."
        # Quick install step now that build is done and pip is free
        { provisioning_install_sageattention 2>&1 | sed 's/^/[SAGE_INSTALL] /'; }
    else
        echo "WARNING: One or more setup tasks failed (Sage: $sage_status, Nodes: $nodes_status)"
    fi

    # Kill monitor loop
    kill $MONITOR_PID 2>/dev/null

    provisioning_print_end "$provisioning_start_time"
}

function provisioning_monitor_loop() {
    local start_time=$(date +%s)
    local interval=10

    # Wait a bit for folders to be created
    sleep 5

    while true; do
        if [[ -d "${COMFYUI_DIR}/models" ]]; then
            local current_bytes=$(du -sb "${COMFYUI_DIR}/models" 2>/dev/null | awk '{print $1}')
        else
            local current_bytes=0
        fi

        [[ -z "$current_bytes" ]] && current_bytes=0

        local now=$(date +%s)
        local elapsed=$((now - start_time))
        [[ $elapsed -eq 0 ]] && elapsed=1

        local percent=0
        if [[ $TOTAL_BYTES_TO_DOWNLOAD -gt 0 ]]; then
            percent=$((current_bytes * 100 / TOTAL_BYTES_TO_DOWNLOAD))
        fi

        local speed=0
        speed=$((current_bytes / elapsed))

        local eta=0
        local eta_msg=""

        local download_eta=0
        local remaining_bytes=$((TOTAL_BYTES_TO_DOWNLOAD - current_bytes))
        if [[ $speed -gt 0 ]]; then
            download_eta=$((remaining_bytes / speed))
        fi

        local setup_eta=$((MIN_SETUP_TIME - elapsed))
        [[ $setup_eta -lt 0 ]] && setup_eta=0

        if [[ $download_eta -ge $setup_eta ]]; then
            eta=$download_eta
        else
            eta=$setup_eta
            eta_msg=" (Setup)"
        fi

        local current_gb=$(echo "scale=2; $current_bytes/1024/1024/1024" | bc 2>/dev/null || echo "0")
        local total_gb=$(echo "scale=2; $TOTAL_BYTES_TO_DOWNLOAD/1024/1024/1024" | bc 2>/dev/null || echo "0")
        local speed_mb=$(echo "scale=2; $speed/1024/1024" | bc 2>/dev/null || echo "0")
        local eta_min=$((eta / 60))
        local eta_sec=$((eta % 60))
        local elapsed_min=$((elapsed / 60))
        local elapsed_sec=$((elapsed % 60))

        echo -e "\n[PROGRESS] ${current_gb}GB / ${total_gb}GB (${percent}%) | Elapsed: ${elapsed_min}m ${elapsed_sec}s | Speed: ${speed_mb}MB/s | ETA: ${eta_min}m ${eta_sec}s${eta_msg}"

        echo "[PROG_DATA] {\"downloaded_bytes\": $current_bytes, \"total_bytes\": $TOTAL_BYTES_TO_DOWNLOAD, \"percentage\": $percent, \"speed_bps\": $speed, \"eta_seconds\": $eta, \"elapsed_seconds\": $elapsed}"

        sleep $interval
    done
}

function provisioning_get_apt_packages() {
    if [[ -n $APT_PACKAGES ]]; then
        local apt_cmd="${APT_INSTALL:-apt-get install -y}"
        sudo $apt_cmd ${APT_PACKAGES[@]}
    fi
}

function provisioning_get_pip_packages() {
    if [[ -n $PIP_PACKAGES ]]; then
        pip install --no-cache-dir ${PIP_PACKAGES[@]}
    fi
}

function provisioning_build_sageattention() {
    local start_time=$(date +%s)
    echo "Building SageAttention from source..."

    local sage_dir="${WORKSPACE}/SageAttention"

    if [[ ! -d "$sage_dir" ]]; then
        echo "Cloning SageAttention repository..."
        git clone https://github.com/thu-ml/SageAttention.git "$sage_dir"
    else
        echo "SageAttention directory exists, pulling latest..."
        ( cd "$sage_dir" && git pull )
    fi

    export EXT_PARALLEL=4
    export NVCC_APPEND_FLAGS="--threads 8"
    export MAX_JOBS=32

    # Build ONLY for the provisioned host's GPU arch (host-matched = fastest + correct).
    local arch
    arch=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' ')
    if [[ ! $arch =~ ^[0-9]+\.[0-9]+$ ]]; then
        local gpu_name
        gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)
        case "$gpu_name" in
            *5090*|*5080*|*"RTX 50"*|*Blackwell*) arch="12.0" ;;
            *4090*|*4080*|*"RTX 40"*|*L40*|*Ada*)  arch="8.9"  ;;
            *) arch="12.0" ;;
        esac
        echo "compute_cap query unavailable; inferred arch=${arch} from name '${gpu_name}'"
    fi
    export TORCH_CUDA_ARCH_LIST="$arch"
    echo "Building SageAttention for detected GPU arch: TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"

    echo "Compiling SageAttention extensions..."
    ( cd "$sage_dir" && python setup.py build )

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    echo "SageAttention build complete. Duration: $((duration / 60))m $((duration % 60))s"
}

function provisioning_install_sageattention() {
    local start_time=$(date +%s)
    echo "Installing SageAttention..."

    local sage_dir="${WORKSPACE}/SageAttention"

    ( cd "$sage_dir" && python setup.py install --skip-build )

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    echo "SageAttention installation complete. Duration: $((duration / 60))m $((duration % 60))s"
}

function install_requirements() {
    local requirements_file="$1"
    if [[ -e "$requirements_file" ]]; then
        echo "Installing requirements from $requirements_file"
        if command -v uv &> /dev/null; then
            echo "Using uv for dependency installation..."
            if uv pip install --no-cache-dir -r "$requirements_file"; then
                echo "uv pip install succeeded"
                return 0
            else
                echo "uv pip install failed, falling back to regular pip..."
            fi
        fi
        echo "Using regular pip for dependency installation..."
        pip install --no-cache-dir -r "$requirements_file"
    fi
}

function provisioning_get_nodes() {
    local start_time=$(date +%s)
    for repo in "${NODES[@]}"; do
        dir="${repo##*/}"
        path="${COMFYUI_DIR}/custom_nodes/${dir}"
        requirements="${path}/requirements.txt"
        if [[ -d $path ]]; then
            if [[ ${AUTO_UPDATE,,} != "false" ]]; then
                echo "Updating node: ${repo}"
                ( cd "$path" && git pull )
                install_requirements "$requirements"
            fi
        else
            echo "Downloading node: ${repo}"
            git clone "${repo}" "${path}" --recursive
            install_requirements "$requirements"
        fi
    done
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    echo "Nodes installation complete. Duration: $((duration / 60))m $((duration % 60))s"
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
    echo -e "#   LongCat-Avatar-1.5 — ComfyUI provision   #"
    echo -e "#   WanVideoWrapper + distill LoRA + Sage     #"
    echo -e "# Your container will be ready on completion #"
    echo -e "##############################################\\n"
}

function provisioning_print_end() {
    local start_time="$1"
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    echo -e "\\nProvisioning complete: Application will start now"
    echo -e "Total provisioning time: $((duration / 60))m $((duration % 60))s\\n"
    echo -e "NEXT STEPS (in ComfyUI):"
    echo -e "  - Load workflow: LongCatAvatar_audio_image_to_video_example_01.json"
    echo -e "  - Model loader -> LongCat-Avatar-single_fp8_e4m3fn_scaled_mixed_KJ.safetensors"
    echo -e "  - Block swap: 5090=0, 4090=20 (raise to 25 if OOM)"
    echo -e "  - Steps: 8 (distill). Resolution: 832x480 for the 60s/15-30min budget.\\n"
}

function provisioning_download() {
    local url="$1"
    local dir="$2"
    local auth_header=""
    local filename=""

    filename=$(basename "${url%%\?*}")

    # For HuggingFace files, prefer hf_transfer (Rust multi-threaded) via the `hf` CLI.
    if [[ $url =~ ^https://huggingface\.co/(.+)/resolve/([^/]+)/(.+)$ ]]; then
        local repo="${BASH_REMATCH[1]}"
        local revision="${BASH_REMATCH[2]}"
        local path_in_repo="${BASH_REMATCH[3]}"
        local hf_cli=""
        command -v hf &>/dev/null && hf_cli="hf"
        [[ -z $hf_cli ]] && command -v huggingface-cli &>/dev/null && hf_cli="huggingface-cli"
        if [[ -n $hf_cli ]]; then
            if HF_HUB_ENABLE_HF_TRANSFER=1 "$hf_cli" download "$repo" "$path_in_repo" \
                    --revision "$revision" --local-dir "$dir" \
                    ${HF_TOKEN:+--token "$HF_TOKEN"}; then
                # hf preserves repo sub-paths; flatten to the bare filename if needed.
                if [[ "$path_in_repo" != "$filename" && -f "$dir/$path_in_repo" ]]; then
                    mv -f "$dir/$path_in_repo" "$dir/$filename"
                fi
                return 0
            fi
            echo "hf download failed for $url — falling back to aria2c"
        fi
    fi

    if [[ -n $HF_TOKEN && $url =~ ^https://([a-zA-Z0-9_-]+\\.)?huggingface\\.co(/|$|\\?) ]]; then
        auth_header="--header=Authorization: Bearer $HF_TOKEN"
    fi

    if [[ -n $auth_header ]]; then
        aria2c -x 12 -s 12 -k 1M -c --summary-interval=0 --console-log-level=warn \
            --allow-overwrite=true --auto-file-renaming=false --file-allocation=none \
            -o "$filename" $auth_header -d "$dir" "$url"
    else
        aria2c -x 12 -s 12 -k 1M -c --summary-interval=0 --console-log-level=warn \
            --allow-overwrite=true --auto-file-renaming=false --file-allocation=none \
            -o "$filename" -d "$dir" "$url"
    fi
}

if [[ ! -f /.noprovisioning ]]; then
    provisioning_start
fi
