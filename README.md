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
- At least 50GB free disk space for model weights
- Fast internet connection for model downloads

## Next Steps

After successful local testing:
1. Deploy to vast.ai infrastructure
2. Configure production environment
3. Set up monitoring and logging
4. Optimize for performance
