#!/bin/bash
#
# Check download sizes for povision_fp8.sh without downloading entire files
# Downloads ~2 seconds of each file to extract total size from aria2c output
#

urls=(
    "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors"
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors"
    "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"
    "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-bf16.safetensors"
    "https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/InfiniteTalk/Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors"
    "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors"
    "https://huggingface.co/Kijai/MelBandRoFormer_comfy/resolve/6251b3a2bd544aaa31400138e55abda4722735cc/MelBandRoformer_fp16.safetensors"
)

total_mb=0
echo "Checking file sizes (downloading ~100MB total to determine sizes)..."
echo ""

for url in "${urls[@]}"; do
    filename=$(basename "${url%%\?*}")

    # Start download in background, let it run for 2 seconds, then kill it
    tmpfile="/tmp/size_check_$$.log"
    timeout 3 aria2c --summary-interval=1 -x 1 -s 1 "$url" -d /tmp -o "check_$$.partial" 2>&1 > "$tmpfile" &
    pid=$!
    sleep 2
    kill $pid 2>/dev/null
    wait $pid 2>/dev/null

    # Extract size from output
    size=$(grep -oP '\d+(\.\d+)?(MiB|GiB)/\K\d+(\.\d+)?(MiB|GiB)' "$tmpfile" | head -1)

    if [[ -n $size ]]; then
        echo "$filename: $size"

        # Convert to MB for total
        if [[ $size =~ ([0-9.]+)GiB ]]; then
            gb="${BASH_REMATCH[1]}"
            mb=$(echo "$gb * 1024" | bc)
            total_mb=$(echo "$total_mb + $mb" | bc)
        elif [[ $size =~ ([0-9.]+)MiB ]]; then
            mb="${BASH_REMATCH[1]}"
            total_mb=$(echo "$total_mb + $mb" | bc)
        fi
    else
        echo "$filename: Unable to determine"
    fi

    rm -f "$tmpfile" "/tmp/check_$$.partial"
done

echo ""
echo "=========================================="
total_gb=$(echo "scale=2; $total_mb / 1024" | bc)
echo "Total download size: ${total_gb} GB (${total_mb} MB)"
echo ""
