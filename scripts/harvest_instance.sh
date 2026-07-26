#!/usr/bin/env bash
# Pull everything worth keeping off a vast.ai rental before destroying it.
#
# Written after the previous instance was destroyed with its ComfyUI stdout
# unrecoverable: run_matrix.sh restarts truncate /var/log/portal/comfyui.log and
# the .old rotation was empty. Anything not copied here is gone for good, so this
# errs toward taking too much.
#
#   ./harvest_instance.sh 159.48.242.14 41982 45944385
#
# Videos, harness logs, service logs, result JSONs, the built SageAttention
# wheel, the patched node source and an environment snapshot.
set -uo pipefail

HOST="${1:?usage: harvest_instance.sh <host> <port> <instance_id>}"
PORT="${2:?}"
INSTANCE="${3:?}"
DEST="${DEST:-output/vast_${INSTANCE}}"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=20 -p ${PORT}"

mkdir -p "$DEST"/{videos,logs,results,service_logs,nodes,wheel}

say() { echo "[harvest] $*"; }

# Snapshot the environment first: it is cheap and it is the thing that explains
# every number in the results when read back months later.
say "environment snapshot"
timeout 180 ssh $SSH_OPTS "root@${HOST}" '
    echo "## nvidia-smi";        nvidia-smi
    echo; echo "## torch/gpu";   /venv/main/bin/python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.get_device_capability())" 2>&1
    echo; echo "## sage";        /venv/main/bin/python -c "import sageattention,os;print(sageattention.__file__)" 2>&1
    echo; echo "## supervisor";  cat /etc/supervisor/conf.d/comfyui.conf 2>/dev/null
    echo; echo "## disk";        df -h /workspace /
    echo; echo "## model sizes"; du -sh /workspace/ComfyUI/models/* 2>/dev/null
    echo; echo "## inductor";    du -sh /tmp/torchinductor_root 2>/dev/null
    echo; echo "## pip freeze";  /venv/main/bin/pip freeze 2>/dev/null
' > "$DEST/environment.txt" 2>&1
say "  -> $(wc -l <"$DEST/environment.txt") lines"

say "videos"
timeout 900 rsync -az --info=stats1 -e "ssh $SSH_OPTS" \
    "root@${HOST}:/workspace/ComfyUI/output/" "$DEST/videos/" 2>&1 | tail -3

say "harness logs + vram sidecars"
timeout 300 rsync -az -e "ssh $SSH_OPTS" \
    "root@${HOST}:/workspace/logs/" "$DEST/logs/" 2>&1 | tail -2

say "results"
timeout 300 rsync -az -e "ssh $SSH_OPTS" \
    "root@${HOST}:/workspace/results/" "$DEST/results/" 2>&1 | tail -2
timeout 300 rsync -az -e "ssh $SSH_OPTS" \
    "root@${HOST}:/workspace/results_pass1/" "$DEST/results_pass1/" 2>&1 | tail -2

say "workflows actually executed"
timeout 300 rsync -az -e "ssh $SSH_OPTS" \
    "root@${HOST}:/workspace/sweep/" "$DEST/workflows/" 2>&1 | tail -2

say "service logs (provisioning, comfyui, supervisor)"
timeout 300 rsync -az -e "ssh $SSH_OPTS" \
    "root@${HOST}:/var/log/portal/" "$DEST/service_logs/portal/" 2>&1 | tail -2
timeout 300 rsync -az -e "ssh $SSH_OPTS" \
    "root@${HOST}:/var/log/supervisor/" "$DEST/service_logs/supervisor/" 2>&1 | tail -2

say "sage wheel"
timeout 300 rsync -az -e "ssh $SSH_OPTS" \
    "root@${HOST}:/workspace/SageAttention/dist/" "$DEST/wheel/" 2>&1 | tail -2

say "patched node source (both sides of the WANOPT patch)"
NODE=/workspace/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper.git/multitalk/multitalk_loop.py
timeout 120 rsync -az -e "ssh $SSH_OPTS" \
    "root@${HOST}:${NODE}" "$DEST/nodes/multitalk_loop.patched.py" 2>&1 | tail -1
timeout 120 rsync -az -e "ssh $SSH_OPTS" \
    "root@${HOST}:${NODE}.orig" "$DEST/nodes/multitalk_loop.orig.py" 2>&1 | tail -1

say "done -> $DEST"
du -sh "$DEST"
find "$DEST" -type f | wc -l | xargs echo "files:"
