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

Mon Dec  8 22:20:07 UTC 2025
** ComfyUI startup time: 2025-12-08 22:22:31.139

Prompt executed in 00:16:23

overall potentially under 20 minutes e2e for 60s video (on vastai instance with fast internet)

for instance with $0.60/hr this should be less than $0.30 per video