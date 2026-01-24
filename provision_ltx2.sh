#!/bin/bash
# 
# Provisioning script for Vast.ai LTX-2 + Custom Audio ComfyUI environment.
# Sets up ComfyUI, installs custom nodes, downloads models (with progress monitoring), and compiles SageAttention.
#
# Reference Tutorial: https://www.nextdiffusion.ai/tutorials/ltx2-image-to-video-with-custom-audio-comfyui
#
# Total Download Size: ~41.8 GB
# Key Model Sizes:
# - ltx-2-19b-dev-fp8.safetensors: ~25.2 GB
# - MelBandRoformer_fp16.safetensors: ~435 MB
# - ltx-2-19b-distilled-lora-384.safetensors: ~7.1 GB
# - umt5-xxl-enc-fp8_e4m3fn.safetensors: ~6.3 GB
# - ltx-2-19b-ic-lora-detailer.safetensors: ~2.4 GB
# - ltx-2-19b-lora-camera-control-dolly-in.safetensors: ~312 MB
#
# Monitor Output:
# The script provides download progress in two formats every 10 seconds:
# 1. Human Readable: [PROGRESS] <Downloaded>GB / <Total>GB (<Percent>%) | Elapsed: <Min>m <Sec>s | Speed: <Speed>MB/s | ETA: <Min>m <Sec>s
# 2. Machine Readable: [PROG_DATA] JSON_OBJECT
#    JSON Schema: {"downloaded_bytes": int, "total_bytes": int, "percentage": int, "speed_bps": int, "eta_seconds": int, "elapsed_seconds": int}
#

source /venv/main/bin/activate
COMFYUI_DIR=${WORKSPACE}/ComfyUI
# Exact total bytes (LTX2 + Audio + LoRAs + Common encoders)
TOTAL_BYTES_TO_DOWNLOAD=44885798434 # ~41.8 GB
MIN_SETUP_TIME=360  # Minimum 6 minutes for nodes + SageAttention build

APT_PACKAGES=(aria2 bc)
PIP_PACKAGES=(
)

NODES=(
    "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"
    "https://github.com/kijai/ComfyUI-MelBandRoFormer"
    "https://github.com/kijai/ComfyUI-KJNodes"
    "https://github.com/kijai/ComfyUI-Video-Post-Processing"
    "https://github.com/ltdrdata/ComfyUI-Manager"
    "https://github.com/Lightricks/ComfyUI-LTXVideo"
)

# ... (Checkpoints/Models sections remain unchanged) ...

    # Check for ComfyUI update logic (optional but recommended)
    echo "Updating ComfyUI..."
    (cd "${COMFYUI_DIR}" && git pull origin master && pip install -r requirements.txt)


    # Kill monitor loop
    kill $MONITOR_PID 2>/dev/null

    provisioning_print_end "$provisioning_start_time"
}

function provisioning_monitor_loop() {
    local start_time=$(date +%s)
    local interval=10
    local last_bytes=0
    
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
        
        local speed=$((current_bytes / elapsed))
        
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
    
    echo "Compiling SageAttention extensions..."
    ( cd "$sage_dir" && python setup.py build )
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local minutes=$((duration / 60))
    local seconds=$((duration % 60))
    echo "SageAttention build complete. Duration: ${minutes}m ${seconds}s"
}

function provisioning_install_sageattention() {
    local start_time=$(date +%s)
    echo "Installing SageAttention..."
    
    local sage_dir="${WORKSPACE}/SageAttention"
    
    ( cd "$sage_dir" && python setup.py install --skip-build )
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local minutes=$((duration / 60))
    local seconds=$((duration % 60))
    echo "SageAttention installation complete. Duration: ${minutes}m ${seconds}s"
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
    local minutes=$((duration / 60))
    local seconds=$((duration % 60))
    echo "Nodes installation complete. Duration: ${minutes}m ${seconds}s"
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
    local start_time="$1"
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local minutes=$((duration / 60))
    local seconds=$((duration % 60))
    echo -e "\\nProvisioning complete: Application will start now"
    echo -e "Total provisioning time: ${minutes}m ${seconds}s\\n"
}

function provisioning_download() {
    local url="$1"
    local dir="$2"
    local auth_header=""
    local filename=""

    filename=$(basename "${url%%\?*}")

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

if [[ ! -f /.noprovisioning ]]; then
    provisioning_start
fi
