# InfiniteTalk Docker Testing Environment

This project provides a local testing environment for InfiniteTalk Docker setup before deployment to vast.ai infrastructure.

## Project Overview

InfiniteTalk is an AI-powered video generation system that combines:
- Wan2.1-I2V-14B-480P video generation model
- Chinese Wav2Vec2 audio processing
- InfiniteTalk speech-to-video synchronization

This repository contains testing and setup scripts to verify the Docker environment locally before deploying to vast.ai.

## Directory Structure

- `SIMPLE_START.md` - Step-by-step Docker setup guide for testing
- `docker_data/` - Persistent storage for Docker container data
  - Stores downloaded model weights to avoid re-downloading
  - Contains configuration files and test outputs
  - Survives container restarts and recreation
- `InfiniteTalk/` - InfiniteTalk source code and models (gitignored)
- `scripts/` - Utility scripts for vast.ai operations
- `input_files/` - Input files for testing
- `.gitignore` - Git ignore rules
- `README.md` - This file

## Git Setup

This repository uses Git for version control. The InfiniteTalk directory is gitignored to avoid tracking large model files and source code that may be managed separately.

### Ignored Files/Directories
- `InfiniteTalk/` - Complete InfiniteTalk source code and models
- `.hf_home/` - HuggingFace cache directory
- `.venv-backups/` - Virtual environment backups
- `docker_data/` - Docker persistent data (may contain large files)

## Docker Setup

Follow the steps in `SIMPLE_START.md` to:
1. Download and start the vastai/pytorch Docker container
2. Verify pre-installed dependencies (PyTorch, CUDA, etc.)
3. Install additional requirements (xformers, flash-attn, etc.)
4. Download model weights to `docker_data/` directory
5. Test the complete InfiniteTalk pipeline

## Usage

```bash
# Start Docker container (mounts current directory to /workspace)
docker run -it --gpus all \
  -v $(pwd):/workspace \
  vastai/pytorch:2.4.1-cuda-12.4.1-py310-22.04 \
  /bin/bash

# Inside container, follow SIMPLE_START.md steps
cd /workspace
# ... run verification commands
```

## Requirements

- Docker with NVIDIA GPU support
- NVIDIA drivers with CUDA 12.4+ compatibility
- At least 30GB free disk space for model weights (see download sizes below)
- Fast internet connection for model downloads

## Provisioning Scripts

### povision_fp8.sh

Automated provisioning script for ComfyUI with Wan2.1 models in FP8 format.

**Total Download Size: ~29.94 GB**

#### Model File Breakdown:
- `Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors`: 15 GB (main diffusion model)
- `umt5-xxl-enc-bf16.safetensors`: 10 GB (text encoder)
- `Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors`: 2.5 GB
- `clip_vision_h.safetensors`: 1.1 GB
- `lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors`: 703 MB
- `MelBandRoformer_fp16.safetensors`: 435 MB
- `Wan2_1_VAE_bf16.safetensors`: 242 MB

#### ComfyUI Custom Nodes:
- ComfyUI-WanVideoWrapper
- ComfyUI-VideoHelperSuite
- ComfyUI-MelBandRoFormer
- ComfyUI-KJNodes

**Check current sizes:** Run `scripts/check_download_sizes.sh` to verify latest file sizes without downloading entire files.

## Next Steps

After successful local testing:
1. Deploy to vast.ai infrastructure
2. Configure production environment
3. Set up monitoring and logging
4. Optimize for performance

## Quick Start on vast.ai

```bash
# Install aria2 for faster downloads
apt-get update && apt-get install -y aria2

# Download and run provisioning script
wget https://raw.githubusercontent.com/daromaj/vast_experiments/refs/heads/master/povision_fp8.sh
chmod +x povision_fp8.sh
./povision_fp8.sh

# Copy outputs from vast.ai instance
vastai copy INSTANCE_ID:/workspace/ComfyUI/output local:output
```

## execution time for ~60s video

### 2026-07-26 — current numbers (RTX 5090, measured)

**58 s audio (1455 frames, 21 windows) @ 480x832: 9 m 50 s**, down from 13 m 29 s.

The old figures below were bottlenecked by a SageAttention wheel built for the
wrong GPU. It was cached under `python/sage/torch2.10.0-cu128-sm_120/` but
compiled for Ada, so on a 5090 sage fell back silently — correct output, but
>=6 GB more VRAM than SDPA, which OOMed `WanVideoSampler` at 29.3 GiB. Rebuilt
with `TORCH_CUDA_ARCH_LIST=12.0` it is the fastest backend by a wide margin.
A wheel's filename tells you nothing about its target: SageAttention 2.2.0 always
emits `_qattn_sm80` / `_qattn_sm89` because those are the kernel *sources*.

8 s clip, run 2 of 2 (run 1 discarded — it pays the torch.compile warmup):

| Config | Attention | Steps | Gen time | vs baseline |
|---|---|---|---|---|
| sage + 4-step + quick wins + node patch | sageattn | 4 | **72.3 s** | **2.93x** |
| sage + 4-step + quick wins | sageattn | 4 | 81.7 s | 2.59x |
| sage + 4-step distill | sageattn | 4 | 87.7 s | 2.41x |
| sdpa + 4-step distill | sdpa | 4 | 117.8 s | 1.80x |
| old 6-step baseline | sdpa | 6 | 211.5 s | — |

The 9 m 50 s full-clip figure above predates the last two rounds; it was measured
at the 87.7 s config. The 72.3 s config should land nearer 8 minutes but that has
not been measured on a full 58 s clip — do not quote it as if it had been.

Isolated attention kernel at the real shape (40 heads x 32,760 tokens x 128 dim):
**sageattn 39.1 ms vs sdpa 106.9 ms**.

720p (1280x720) does fit on a 5090 at `blocks_to_swap=0` — 27.8 GiB peak with
`tiled_vae=true` — but costs ~4.4x wall-clock, and there is no 720p lightx2v
distill LoRA published (only 480p ranks exist), so the 4-step LoRA is a mismatch
at that resolution. See `july_test.md`.

Full method, per-setting attribution and the source audit behind the quick wins:
`july_test.md` and `notes/node_optimization_audit.md`.

### Disk sizing

Measured on a live 5090 rental after a full provision + several renders:

| Item | Size |
|---|---|
| `/workspace/ComfyUI/models` | 32 GB |
| — `diffusion_models` (Wan 14B fp8 + InfiniteTalk) | 19 GB |
| — `text_encoders` (umt5-xxl) | 11 GB |
| — clip_vision / loras / vae / wav2vec2 / transformers | ~2.7 GB |
| `/venv` | 8.8 GB |
| torch inductor cache (grows with compiles) | 1.6 GB |
| custom_nodes + SageAttention build | 0.6 GB |
| **fixed setup total** | **~43 GB** |

Per generated video the marginal cost is tiny — ComfyUI writes both a silent
`.mp4` and an `-audio.mp4`:

| Videos @ 1 min | Output size | Total disk |
|---|---|---|
| 1 | ~16 MB | ~43 GB |
| 10 | ~160 MB | ~43 GB |

**So `--disk 60` is comfortable and `--disk 50` is workable; the models dominate
and video count is irrelevant at this scale.** Adding the 720P checkpoint would
push the fixed cost to ~60 GB, so budget `--disk 80` if you provision both.

### older figures (pre-fix, kept for reference)

Mon Dec  8 22:20:07 UTC 2025
** ComfyUI startup time: 2025-12-08 22:22:31.139

Prompt executed in 00:16:23

overall potentially under 20 minutes e2e for 60s video (on vastai instance with fast internet)

for instance with $0.60/hr this should be less than $0.30 per video

13:29 for 58s audio with sageattention 2 and 3 installed but regular sage attention selected

8 minutes just to get comfyui up and running

4090 - 314s for 10s video with sageattention 2 and block swap 20 ~ 32 minutes for 60s video
4090 - 238s for 10s video with sageattention 2 and block swap 5 ~ 24minutes for 60s video

58s Prompt executed in 00:49:55

we need to choose pcie4 for 4090
so far I was not able to fit models in 4090 memory

Also - if the host is not ready within a minute it's probably better to cancel the instance and try different one

4090
# Run this inside a standard PyTorch 2.5/CUDA 12.4 container
export TORCH_CUDA_ARCH_LIST="8.9"
python setup.py bdist_wheel

5090
# Run this on your CUDA 12.9 machine
export TORCH_CUDA_ARCH_LIST="12.0"

# Optional: Add a local version tag so you don't mix them up
export SAG_VERSION_SUFFIX="+cu129" 

python setup.py bdist_wheel

export TORCH_CUDA_ARCH_LIST="12.0"
export SAG_VERSION_SUFFIX="+cu129"

# Instead of "python setup.py bdist_wheel", use:
pip wheel . --no-deps -w dist/



---
# audio 5s

## 5090 PCIE 4.0/16x
sage attention 2 (~6-7s per it)
<102s
flash attention (~8-9s per it)
<116s

# audio 58s - 21 windows

## 5090

sage attention 2 (~6-7s per it) 34s per window
~12m
flash attention (~8-9s per it) 49s per window + ~5s rest
18:20

## 4090 PCIE 4.0/16x
sage attention 2 (~7-12s per it) 46s per window
18:45

flash attention (~11-15s per it) 1:09 per window + ~15s rest
26:14
~24:30

## 5090
[FLASH] Flash Attention installation complete. Duration: 0m 19s
[SAGE_INSTALL] SageAttention installation complete. Duration: 2m 51s

## 4090
[FLASH] Flash Attention installation complete. Duration: 3m 42s
[FLASH] Flash Attention installation complete. Duration: 0m 36s
[SAGE_BUILD] SageAttention build complete. Duration: 4m 32s
[SAGE_INSTALL] SageAttention installation complete. Duration: 0m 3s

--- download examples
[PROGRESS] 37.73GB / 37.73GB (100%) | Elapsed: 2m 5s | Speed: 309.09MB/s | ETA: 3m 55s (Setup)

-- download around 100Mb/s
[PROGRESS] 37.73GB / 37.73GB (100%) | Elapsed: 7m 47s | Speed: 82.73MB/s | ETA: 0m 0s
[PROGRESS] 37.73GB / 37.73GB (100%) | Elapsed: 8m 47s | Speed: 73.31MB/s | ETA: 0m 0s


++ instance creation time !!! 1m is acceptable and expected
total expected time - around 25 minutes for 4090 and 20m for 5090
