#!/bin/bash
#
# Settle one question: is the download ceiling the host's shared uplink, or a
# per-transfer cap on HuggingFace's side?
#
# It decides whether provisioning_get_files should keep downloading one file at a
# time. Against a shared-link cap, parallel files gain nothing. Against a
# per-transfer cap, they multiply throughput for free.
#
# Run ON the instance. Downloads to /tmp and cleans up after itself.
#
set -u
TMP=$(mktemp -d /tmp/bwtest.XXXXXX)
trap 'rm -rf "$TMP"' EXIT

HF=https://huggingface.co/Kijai/WanVideo_comfy/resolve/main
A="${HF}/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"  # ~703MB
B="${HF}/Wan2_1_VAE_bf16.safetensors"                                              # ~242MB
C=https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors  # ~1.18GB

# Same flags the provisioning script uses, so the numbers transfer directly.
ARIA="aria2c -x 12 -s 12 -k 1M --summary-interval=0 --console-log-level=error \
      --allow-overwrite=true --auto-file-renaming=false --file-allocation=none"

rate() {  # rate <bytes> <ms>
    echo $(( $1 * 1000 / $2 / 1024 / 1024 ))
}

echo "=========================================================="
echo " TEST 1: single file, sequential (what we do today)"
echo "=========================================================="
t0=$(date +%s%3N)
$ARIA -d "$TMP" -o a1.bin "$A" >/dev/null 2>&1
t1=$(date +%s%3N)
sz=$(stat -c %s "$TMP/a1.bin" 2>/dev/null || echo 0)
single=$(rate "$sz" $((t1 - t0)))
echo "  $((sz/1024/1024))MB in $(( (t1-t0)/1000 ))s  =>  ${single} MB/s"
rm -f "$TMP"/*.bin

echo
echo "=========================================================="
echo " TEST 2: three files concurrently"
echo "=========================================================="
t0=$(date +%s%3N)
$ARIA -d "$TMP" -o p1.bin "$A" >/dev/null 2>&1 &
$ARIA -d "$TMP" -o p2.bin "$B" >/dev/null 2>&1 &
$ARIA -d "$TMP" -o p3.bin "$C" >/dev/null 2>&1 &
wait
t1=$(date +%s%3N)
sz=$(du -cb "$TMP"/p*.bin 2>/dev/null | tail -1 | awk '{print $1}')
par=$(rate "$sz" $((t1 - t0)))
echo "  $((sz/1024/1024))MB in $(( (t1-t0)/1000 ))s  =>  ${par} MB/s aggregate"
rm -f "$TMP"/*.bin

echo
echo "=========================================================="
echo " TEST 3: non-HF control (Cloudflare, 1GB)"
echo "=========================================================="
t0=$(date +%s%3N)
curl -s -o "$TMP/cf.bin" "https://speed.cloudflare.com/__down?bytes=1073741824" 2>/dev/null
t1=$(date +%s%3N)
sz=$(stat -c %s "$TMP/cf.bin" 2>/dev/null || echo 0)
if [[ $sz -gt 1000000 ]]; then
    cf=$(rate "$sz" $((t1 - t0)))
    echo "  $((sz/1024/1024))MB in $(( (t1-t0)/1000 ))s  =>  ${cf} MB/s"
else
    cf=0
    echo "  control unavailable (blocked or failed)"
fi

echo
echo "=========================================================="
echo " VERDICT"
echo "=========================================================="
echo "  sequential single file : ${single} MB/s"
echo "  3 files in parallel    : ${par} MB/s"
[[ ${cf:-0} -gt 0 ]] && echo "  non-HF control         : ${cf} MB/s"
echo
# A shared uplink caps the TOTAL, so parallelism cannot beat it by much. A
# per-transfer cap leaves the total free, so parallelism scales close to linearly.
if [[ $par -gt $(( single * 15 / 10 )) ]]; then
    echo "  => PER-TRANSFER CAP. Parallel downloads win (${single} -> ${par} MB/s)."
    echo "     Downloading files concurrently is worth implementing."
else
    echo "  => SHARED-LINK CAP (~${single} MB/s ceiling)."
    echo "     Parallelism will not help; the payload itself has to shrink."
fi
