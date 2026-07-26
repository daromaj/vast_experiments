#!/bin/bash
# 
# Provisioning script for Vast.ai FP8 ComfyUI environment.
# Sets up ComfyUI, installs custom nodes, downloads models (with progress monitoring), and compiles SageAttention.
#
# Total Download Size: ~37.7 GB
# Key Model Sizes:
# - Wan2_1_VAE_bf16.safetensors: ~242 MB
# - clip_vision_h.safetensors: ~1.18 GB
# - lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors: ~703 MB
# - umt5-xxl-enc-bf16.safetensors: ~10.58 GB
# - umt5-xxl-enc-fp8_e4m3fn.safetensors: ~6.27 GB
# - Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors: ~2.53 GB
# - Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors: ~15.83 GB
# - MelBandRoformer_fp16.safetensors: ~435 MB
# - wav2vec2-chinese-base_fp16.safetensors: ~190 MB (models/wav2vec2, Wav2VecModelLoader)
#
# Monitor Output:
# The script provides download progress in two formats every 10 seconds:
# 1. Human Readable: [PROGRESS] <Downloaded>GB / <Total>GB (<Percent>%) | Elapsed: <Min>m <Sec>s | Speed: <Speed>MB/s | ETA: <Min>m <Sec>s
# 2. Machine Readable: [PROG_DATA] JSON_OBJECT
#    JSON Schema: {"downloaded_bytes": int, "total_bytes": int, "percentage": int, "speed_bps": int, "eta_seconds": int, "elapsed_seconds": int}
#

source /venv/main/bin/activate
COMFYUI_DIR=${WORKSPACE}/ComfyUI
# Dropped unused umt5 fp8 encoder (~6.73 GB) — see TEXT_ENCODERS note.
# +wav2vec (~190 MB) pre-fetch → see WAV2VEC_MODELS note.
TOTAL_BYTES_TO_DOWNLOAD=33978384650
MIN_SETUP_TIME=360  # Minimum 6 minutes for nodes + SageAttention build

APT_PACKAGES=(aria2 bc libcusparse-dev-12-9 libcublas-dev-12-9 libcusolver-dev-12-9 libcufft-dev-12-9 libcurand-dev-12-9)
PIP_PACKAGES=(
    # hf_transfer (Rust multi-threaded downloader) + the `hf` CLI for HuggingFace pulls.
    "huggingface_hub[hf_transfer]"
)
# Custom nodes are cloned at HEAD, so every rental can get a different version.
# That has already bitten us twice: the wrapper deleted DownloadAndLoadWav2VecModel
# (see the WAV2VEC_MODELS note below), and it is the same class of drift as the
# floating cuda-12.9-auto image tag that used to invalidate the SageAttention
# wheel. Set NODE_PINS to freeze a repo at a known-good commit when a rental has
# to reproduce an earlier result exactly.
#
# Measured 2026-07-26 on an RTX 5090 (31.36GiB): pinning the wrapper back to
# 339e0fe (2026-01-23, the last commit before a known-good run) did NOT fix the
# WanVideoSampler OOM these workflows now hit, so wrapper drift is not the cause
# of that. The pin is kept for reproducibility, not as a workaround.
NODES=(
    "https://github.com/kijai/ComfyUI-WanVideoWrapper.git"
    "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"
    "https://github.com/kijai/ComfyUI-MelBandRoFormer"
    "https://github.com/kijai/ComfyUI-KJNodes"
)

# repo-basename -> commit-ish. Empty by default: pin only when you need to.
declare -A NODE_PINS=(
    # ["ComfyUI-WanVideoWrapper.git"]="339e0fe"
)

WORKFLOWS=(
    # "https://raw.githubusercontent.com/vast-ai/base-image/refs/heads/main/derivatives/pytorch/derivatives/comfyui/workflows/text_to_video_wan.json"
    "https://raw.githubusercontent.com/daromaj/vast_experiments/refs/heads/master/InfiniteTalk-I2V-FP8-Lip-Sync.json"
    "https://raw.githubusercontent.com/daromaj/vast_experiments/refs/heads/master/workflows/InfiniteTalk-I2V-FP8-Lip-Sync_5090_sage_new_prompts.json"
    # July 2026 optimization candidates (fp8_e4m3fn_fast + merged LoRA + max-autotune-no-cudagraphs)
    # 5090 (32GB): blocks_to_swap=0. 4090 (24GB): blocks_to_swap=20 + prefetch=1 to fit VRAM.
    "https://raw.githubusercontent.com/daromaj/vast_experiments/refs/heads/master/workflows/IT_5090_july2026_4step.json"
    "https://raw.githubusercontent.com/daromaj/vast_experiments/refs/heads/master/workflows/IT_5090_july2026_5step.json"
    "https://raw.githubusercontent.com/daromaj/vast_experiments/refs/heads/master/workflows/IT_4090_july2026_4step.json"
    "https://raw.githubusercontent.com/daromaj/vast_experiments/refs/heads/master/workflows/IT_4090_july2026_5step.json"
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
    # Only the bf16 encoder is loaded by every workflow. The fp8 encoder (6.27 GB) was
    # downloaded but never referenced by any workflow JSON — dropped to save ~6 GB / provision.
    "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-bf16.safetensors"
)

WAV2VEC_MODELS=(
    # Pre-fetch wav2vec so the first generation doesn't stall on a runtime download.
    # Current WanVideoWrapper uses the Wav2VecModelLoader node (reads ComfyUI/models/wav2vec2,
    # config baked into the node). NOTE: existing workflows here reference the OLD
    # DownloadAndLoadWav2VecModel node, which no longer exists in the wrapper — swap that node for
    # "Wav2vec2 Model Loader" and point it at this file. ~190 MB.
    "https://huggingface.co/Kijai/wav2vec2_safetensors/resolve/main/wav2vec2-chinese-base_fp16.safetensors"
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

# Prebuilt wheels are stored under an ABI-keyed DIRECTORY:
#
#   python/sage/torch<ver>-cu<cuda>-sm_<arch>/<wheel>.whl
#
# The ABI cannot go in the filename: PEP 427 requires
# {name}-{version}(-{build})?-{pytag}-{abitag}-{platform}.whl and a build tag must
# start with a digit, so "sageattention-2.2.0-torch2.10.0-...whl" is rejected by
# pip outright. The directory carries the ABI; the filename stays valid.
#
# Keying on the exact triple (not just the GPU arch) is the whole fix: a wheel is
# only loadable against the torch it was linked to, so after an image bump the
# lookup simply misses and we go straight to a source build instead of installing
# something that will fail at import.
#
# Verified working: torch2.10.0-cu128-sm_120 on vastai/comfy:v0.28.0-cuda-12.9-py312
# (probe passed, cosine 0.9993 vs SDPA, RTX 5090).
SAGE_WHEEL_BASE="https://github.com/daromaj/vast_experiments/raw/master/python/sage"
SAGE_WHEEL_FILE="sageattention-2.2.0-cp312-cp312-linux_x86_64.whl"

# Viability probe: import + real GPU attention call + cosine check vs SDPA.
# Exit 0 means the installed SageAttention is genuinely usable, which is the only
# signal we trust before skipping the source build.
SAGE_PROBE_URL="https://raw.githubusercontent.com/daromaj/vast_experiments/refs/heads/master/scripts/sage_abi_probe.py"

PROVISION_T0=$(date +%s)

function phase() {
    # Timestamped marker on a single timeline. Everything runs concurrently here,
    # so absolute offsets are the only way to see what the critical path really is.
    local now=$(date +%s)
    local elapsed=$((now - PROVISION_T0))
    printf '[PHASE] +%dm%02ds %s\n' $((elapsed / 60)) $((elapsed % 60)) "$*"
}

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
    phase "apt install start"
    provisioning_get_apt_packages
    phase "apt done, pip base start"
    provisioning_get_pip_packages
    phase "pip base done"



    # Start Download Monitoring in background
    provisioning_monitor_loop &
    MONITOR_PID=$!

    # Start Parallel Operations
    # 1. Build SageAttention (CPU intensive, no pip lock)
    # 2. Setup Nodes (Network/Disk intensive, locks pip)
    echo "Starting parallel setup: SageAttention Build + Node Installation..."
    phase "parallel setup launched (sage build + nodes), downloads starting"
    
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
    provisioning_get_files \
        "${COMFYUI_DIR}/models/loras" \
        "${LORAS[@]}"    

    provisioning_get_files "${COMFYUI_DIR}/models/clip_vision" "${CLIP_VISION[@]}"
    provisioning_get_files "${COMFYUI_DIR}/models/wav2vec2" "${WAV2VEC_MODELS[@]}"

    phase "downloads finished"

    # Nodes are waited on FIRST, deliberately. Their requirements.txt files are what
    # can move torch out from under us, and a wheel judged before torch settles is a
    # wheel that breaks later — the likeliest explanation for "wheels are unreliable".
    echo "Waiting for node installation to complete..."
    wait $NODES_PID
    local nodes_status=$?
    phase "nodes install finished (status=${nodes_status})"

    # Race the prebuilt wheel against the still-running source build. The build was
    # going to happen anyway, so trying the wheel costs ~30s and, when it works,
    # removes the ~6min build from the critical path. Worst case we lose those 30s.
    local sage_ready=1
    if provisioning_try_sage_wheel; then
        sage_ready=0
        phase "SAGE: wheel PASSED probe — cancelling source build"
        kill "$SAGE_PID" 2>/dev/null
        # The subshell dies instantly; its nvcc children do not. Reap them so they
        # stop stealing CPU from the first generation.
        pkill -f "setup.py build" 2>/dev/null
        wait "$SAGE_PID" 2>/dev/null
    else
        phase "SAGE: wheel unusable — falling back to source build"
        # Leave nothing half-installed for the source install to trip over.
        pip uninstall -y sageattention 2>/dev/null
        wait "$SAGE_PID"
        local sage_status=$?
        phase "SAGE: source build finished (status=${sage_status})"
        if [[ $sage_status -eq 0 ]]; then
            { provisioning_install_sageattention 2>&1 | sed 's/^/[SAGE_INSTALL] /'; }
            sage_ready=0
            provisioning_harvest_sage_wheel
        else
            echo "WARNING: SageAttention source build failed (status=${sage_status})"
        fi
    fi

    if [[ $nodes_status -ne 0 ]]; then
        echo "WARNING: Node installation reported failure (status=${nodes_status})"
    fi

    if [[ $sage_ready -eq 0 ]]; then
        phase "SAGE: READY"
    else
        phase "SAGE: UNAVAILABLE — workflows using sageattn will fail"
    fi

    provisioning_report_disk

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
        # Calculate current size of models directory (where most large files go)
        # Using -s for summary, -b for bytes. 
        # Check specific directories to be accurate or just ComfyUI/models
        # Models are scattered, but mostly in models/
        if [[ -d "${COMFYUI_DIR}/models" ]]; then
            local current_bytes=$(du -sb "${COMFYUI_DIR}/models" 2>/dev/null | awk '{print $1}')
        else
            local current_bytes=0
        fi
        
        # Default to 0 if du fails
        [[ -z "$current_bytes" ]] && current_bytes=0

        # Calculate metrics
        local now=$(date +%s)
        local elapsed=$((now - start_time))
        [[ $elapsed -eq 0 ]] && elapsed=1
        
        local percent=0
        if [[ $TOTAL_BYTES_TO_DOWNLOAD -gt 0 ]]; then
            percent=$((current_bytes * 100 / TOTAL_BYTES_TO_DOWNLOAD))
        fi
        
        # Calculate Speed (bytes per second)
        # Using simple average over total time for stability, or could do delta
        # Let's do delta for more real-time feel
        local speed=0
        # For simplicity in bash, just use average speed so far to avoid jumpiness
        speed=$((current_bytes / elapsed))
        
        # Calculate ETA
        local eta=0
        local eta_msg=""
        
        # 1. Download ETA
        local download_eta=0
        local remaining_bytes=$((TOTAL_BYTES_TO_DOWNLOAD - current_bytes))
        if [[ $speed -gt 0 ]]; then
            download_eta=$((remaining_bytes / speed))
        fi
        
        # 2. Setup ETA (minimum run time check)
        local setup_eta=$((MIN_SETUP_TIME - elapsed))
        [[ $setup_eta -lt 0 ]] && setup_eta=0
        
        # 3. Final ETA is max of both
        if [[ $download_eta -ge $setup_eta ]]; then
            eta=$download_eta
        else
            eta=$setup_eta
            eta_msg=" (Setup)"
        fi
        
        # Format for display
        local current_gb=$(echo "scale=2; $current_bytes/1024/1024/1024" | bc 2>/dev/null || echo "0")
        local total_gb=$(echo "scale=2; $TOTAL_BYTES_TO_DOWNLOAD/1024/1024/1024" | bc 2>/dev/null || echo "0")
        local speed_mb=$(echo "scale=2; $speed/1024/1024" | bc 2>/dev/null || echo "0")
        local eta_min=$((eta / 60))
        local eta_sec=$((eta % 60))
        local elapsed_min=$((elapsed / 60))
        local elapsed_sec=$((elapsed % 60))

        # Human Friendly Output
        echo -e "\n[PROGRESS] ${current_gb}GB / ${total_gb}GB (${percent}%) | Elapsed: ${elapsed_min}m ${elapsed_sec}s | Speed: ${speed_mb}MB/s | ETA: ${eta_min}m ${eta_sec}s${eta_msg}"
        
        # Machine Friendly Output (JSON)
        echo "[PROG_DATA] {\"downloaded_bytes\": $current_bytes, \"total_bytes\": $TOTAL_BYTES_TO_DOWNLOAD, \"percentage\": $percent, \"speed_bps\": $speed, \"eta_seconds\": $eta, \"elapsed_seconds\": $elapsed}"

        sleep $interval
    done
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

function provisioning_install_sageattention_deprecated() {
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

function provisioning_build_sageattention() {
    local start_time=$(date +%s)
    # Builds SageAttention from source with parallel compilation
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

    # Build ONLY for the provisioned host's GPU arch. Each arch in TORCH_CUDA_ARCH_LIST
    # makes NVCC recompile every kernel again (~2x per arch), and sm_120 code won't even
    # load on a 4090 (sm_89) — so single, host-matched arch is both fastest and correct.
    local arch
    arch=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' ')
    if [[ ! $arch =~ ^[0-9]+\.[0-9]+$ ]]; then
        # Older driver without compute_cap query — infer from the GPU name.
        local gpu_name
        gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)
        case "$gpu_name" in
            *5090*|*5080*|*"RTX 50"*|*Blackwell*) arch="12.0" ;;
            *4090*|*4080*|*"RTX 40"*|*L40*|*Ada*)  arch="8.9"  ;;
            *) arch="12.0" ;;  # default to the primary target (5090)
        esac
        echo "compute_cap query unavailable; inferred arch=${arch} from name '${gpu_name}'"
    fi
    export TORCH_CUDA_ARCH_LIST="$arch"
    echo "Building SageAttention for detected GPU arch: TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
    
    # Build SageAttention (Compiles C++/CUDA extensions)
    echo "Compiling SageAttention extensions..."
    ( cd "$sage_dir" && python setup.py build )
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local minutes=$((duration / 60))
    local seconds=$((duration % 60))
    echo "SageAttention build complete. Duration: ${minutes}m ${seconds}s"
}

function provisioning_report_disk() {
    # --disk is currently guessed at (80GB) rather than measured. Record what the
    # box ACTUALLY consumed so the next size is a decision instead of a guess:
    # over-provisioning burns storage $/hr, under-provisioning kills the run at 90%.
    echo "[DISK] --- actual consumption ---"

    # NOT df: on Vast the container sees the HOST overlay (a 3.6TB filesystem),
    # not the --disk allocation, so df reports numbers that have nothing to do
    # with the quota we are paying for. du of the workspace is the real figure.
    local used_bytes
    used_bytes=$(du -sb "${WORKSPACE}" 2>/dev/null | awk '{print $1}')
    [[ -z $used_bytes ]] && used_bytes=0
    # LC_ALL=C or awk's %.1f emits a decimal comma under a European locale, which
    # makes the [DISK_DATA] line invalid JSON and silently unparseable.
    LC_ALL=C awk -v b="$used_bytes" 'BEGIN {
        printf "[DISK] workspace used=%.1fGB (df is the host overlay on Vast, not the --disk quota)\n", b/1073741824
        printf "[DISK_DATA] {\"used_gb\": %.1f}\n", b/1073741824
    }'

    # Per-directory, so we can see whether trimming models or the build tree pays.
    # LC_ALL=C: du renders "4.8M" or "4,8M" depending on locale, and the log should
    # not change shape based on which host we landed on.
    local d
    for d in "${COMFYUI_DIR}/models" "${COMFYUI_DIR}/custom_nodes" "${WORKSPACE}/SageAttention" "${WORKSPACE}/wheels"; do
        [[ -d $d ]] || continue
        echo "[DISK] $(LC_ALL=C du -sh "$d" 2>/dev/null | awk '{print $1}')	${d}"
    done

    # The SageAttention build tree is pure scratch once the package is installed.
    # Not deleted automatically here - measure first, decide after a real run.
    if [[ -d "${WORKSPACE}/SageAttention/build" ]]; then
        echo "[DISK] $(LC_ALL=C du -sh "${WORKSPACE}/SageAttention/build" 2>/dev/null | awk '{print $1}')	<- build scratch, reclaimable"
    fi
}

function provisioning_try_sage_wheel() {
    # Install the arch-matched prebuilt wheel and prove it actually works before we
    # let it replace a ~6 minute source build. Returns 0 only on a clean probe.
    local start_time=$(date +%s)

    # The full ABI triple, read from the torch that is actually installed.
    local abi_tag
    abi_tag=$(python3 -c "
import torch
c = torch.cuda.get_device_capability(0)
print('torch%s-cu%s-sm_%d%d' % (torch.__version__.split('+')[0],
                                (torch.version.cuda or 'none').replace('.', ''),
                                c[0], c[1]))" 2>/dev/null)
    if [[ -z $abi_tag ]]; then
        echo "[SAGE_WHEEL] Could not determine torch/GPU ABI — skipping wheel attempt."
        return 1
    fi

    local wheel_url="${SAGE_WHEEL_BASE}/${abi_tag}/${SAGE_WHEEL_FILE}"
    echo "[SAGE_WHEEL] ABI ${abi_tag} -> ${wheel_url}"

    local wheel_dir="${WORKSPACE}/wheels"
    mkdir -p "$wheel_dir"
    provisioning_download "$wheel_url" "$wheel_dir"

    local wheel_file="${wheel_dir}/$(basename "$wheel_url")"
    if [[ ! -f $wheel_file ]]; then
        echo "[SAGE_WHEEL] No wheel cached for this ABI — source build it is."
        return 1
    fi
    # A 404 can still leave a file on disk containing GitHub's error page. A real
    # wheel is a zip archive ("PK") and is megabytes, not kilobytes.
    if [[ $(stat -c %s "$wheel_file" 2>/dev/null || echo 0) -lt 1000000 ]] ||
       [[ $(head -c 2 "$wheel_file") != "PK" ]]; then
        echo "[SAGE_WHEEL] Downloaded file is not a wheel (404 page?) — source build it is."
        rm -f "$wheel_file"
        return 1
    fi

    # --no-deps is load-bearing: the wheel must never drag in its own torch and
    # invalidate the exact ABI we just waited for the nodes to settle.
    if ! pip install --no-cache-dir --force-reinstall --no-deps "$wheel_file"; then
        echo "[SAGE_WHEEL] pip install failed."
        return 1
    fi

    provisioning_download "$SAGE_PROBE_URL" "$WORKSPACE"
    local probe="${WORKSPACE}/sage_abi_probe.py"
    if [[ ! -f $probe ]]; then
        echo "[SAGE_WHEEL] Probe unavailable — refusing to trust an unverified wheel."
        return 1
    fi

    python3 "$probe" 2>&1 | sed 's/^/[SAGE_PROBE] /'
    local verdict=${PIPESTATUS[0]}

    local duration=$(( $(date +%s) - start_time ))
    echo "[SAGE_WHEEL] Attempt took ${duration}s, probe verdict=${verdict} (0=usable)"
    return "$verdict"
}

function provisioning_harvest_sage_wheel() {
    # A source build that works on this base image is worth keeping. Package it and
    # print the path plus the ABI-keyed name to commit it under, so the next run can
    # match a wheel by filename instead of guessing at it.
    local sage_dir="${WORKSPACE}/SageAttention"
    echo "[SAGE_HARVEST] Packaging the working build as a wheel..."
    if ( cd "$sage_dir" && python setup.py bdist_wheel --skip-build ); then
        echo "[SAGE_HARVEST] Wheel(s) written to ${sage_dir}/dist:"
        ls -la "${sage_dir}/dist"/*.whl 2>/dev/null
        # Print the exact destination path so the wheel lands where the ABI-keyed
        # lookup above will find it on the next rental.
        local abi_tag
        abi_tag=$(python3 -c "
import torch
c = torch.cuda.get_device_capability(0)
print('torch%s-cu%s-sm_%d%d' % (torch.__version__.split('+')[0],
                                (torch.version.cuda or 'none').replace('.', ''),
                                c[0], c[1]))" 2>/dev/null)
        echo "[SAGE_HARVEST] Commit it as: python/sage/${abi_tag}/${SAGE_WHEEL_FILE}"
        echo "[SAGE_HARVEST] e.g.  scripts/vast.sh pull ${sage_dir}/dist/<wheel> ."
        echo "[SAGE_HARVEST] scp this back and commit it before the instance dies."
    else
        echo "[SAGE_HARVEST] bdist_wheel failed (is the 'wheel' package installed?) — nothing to harvest."
    fi
}

function provisioning_install_sageattention() {
    local start_time=$(date +%s)
    echo "Installing SageAttention..."
    
    local sage_dir="${WORKSPACE}/SageAttention"
    
    # Install the pre-built package
    ( cd "$sage_dir" && python setup.py install --skip-build )
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local minutes=$((duration / 60))
    local seconds=$((duration % 60))
    echo "SageAttention installation complete. Duration: ${minutes}m ${seconds}s"
}

function provisioning_install_flash_attn() {
    local start_time=$(date +%s)
    echo "Installing flash-attn..."
    pip install flash-attn --no-build-isolation
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local minutes=$((duration / 60))
    local seconds=$((duration % 60))
    echo "Flash Attention installation complete. Duration: ${minutes}m ${seconds}s"
}

function install_requirements() {
    local requirements_file="$1"
    if [[ -e "$requirements_file" ]]; then
        echo "Installing requirements from $requirements_file"
        # Try uv pip install first
        if command -v uv &> /dev/null; then
            echo "Using uv for dependency installation..."
            if uv pip install --no-cache-dir -r "$requirements_file"; then
                echo "uv pip install succeeded"
                return 0
            else
                echo "uv pip install failed, falling back to regular pip..."
            fi
        fi
        # Fall back to regular pip
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
        pin="${NODE_PINS[$dir]:-}"
        if [[ -d $path ]]; then
            if [[ ${AUTO_UPDATE,,} != "false" && -z $pin ]]; then
                echo "Updating node: ${repo}"
                ( cd "$path" && git pull )
                install_requirements "$requirements"
            fi
        else
            echo "Downloading node: ${repo}"
            git clone "${repo}" "${path}" --recursive
            install_requirements "$requirements"
        fi
        # A pinned node must land on that exact commit whether it was just
        # cloned or already present from a previous run.
        if [[ -n $pin ]]; then
            echo "Pinning ${dir} to ${pin}"
            ( cd "$path" && git fetch --depth 50 origin "$pin" 2>/dev/null || git fetch origin
              git checkout -f "$pin" ) || echo "WARNING: could not pin ${dir} to ${pin}"
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
        # Millisecond resolution: the small files finish in well under a second, and
        # integer seconds would report them as an absurd 0 MB/s.
        local file_start=$(date +%s%3N)
        provisioning_download "$url" "$dir"
        # Per-file timing + achieved throughput. Aggregate speed hides the one slow
        # file that actually sets the download critical path.
        local name=$(basename "${url%%\?*}")
        local ms=$(( $(date +%s%3N) - file_start ))
        [[ $ms -le 0 ]] && ms=1
        local bytes=$(stat -c %s "${dir}/${name}" 2>/dev/null || echo 0)
        local mbps=$(( bytes * 1000 / ms / 1024 / 1024 ))
        # %03d on the fractional part: without padding, 45007ms prints as "45.7s".
        printf '[DL_TIME] %s %d.%03ds %dMB %dMB/s\n' \
            "$name" $((ms / 1000)) $((ms % 1000)) $((bytes / 1024 / 1024)) "$mbps"
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

    # Extract filename from URL (remove query parameters and get last path segment)
    filename=$(basename "${url%%\?*}")

    # For HuggingFace files, prefer hf_transfer (Rust multi-threaded) via the `hf` CLI.
    # Roughly on par with aria2c -x12 on a fast host, but more resilient to HF
    # per-connection throttling. Falls back to aria2c below on any failure.
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

    # Detect HuggingFace URLs and add auth if token exists
    if [[ -n $HF_TOKEN && $url =~ ^https://([a-zA-Z0-9_-]+\\.)?huggingface\\.co(/|$|\\?) ]]; then
        auth_header="--header=Authorization: Bearer $HF_TOKEN"
    fi

    # aria2c fallback / non-HF URLs (12 parallel connections, auto-resume)
    # --file-allocation=none: Required for accurate disk usage monitoring during download
    # --summary-interval=0: Suppress aria2c native progress summary to avoid clutter
    if [[ -n $auth_header ]]; then
        aria2c -x 12 -s 12 -k 1M -c --summary-interval=0 --console-log-level=warn \
            --allow-overwrite=true --auto-file-renaming=false --file-allocation=none \
            -o "$filename" $auth_header -d "$dir" "$url"
    else
        aria2c -x 12 -s 12 -k 1M -c --summary-interval=0 --console-log-level=warn \
            --allow-overwrite=true --auto-file-renaming=false --file-allocation=none \
            -o "$filename" -d "$dir" "$url"
    fi

    # Note: No explicit error handling - continue on failures, check logs later
}

if [[ ! -f /.noprovisioning ]]; then
    provisioning_start
fi
