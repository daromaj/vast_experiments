# InfiniteTalk Docker Setup Plan

This document outlines the step-by-step process for setting up InfiniteTalk in a Docker environment using the `vastai/pytorch:2.4.1-cuda-12.4.1-py310-22.04` image. The goal is to create a reliable setup that can be replicated on vast.ai.

## Important Note
We're going to test these steps one by one in the Docker container to verify:
- What's already pre-installed in the vastai/pytorch image
- If we need additional steps or can skip some
- Any dependency conflicts or missing packages
- Performance and compatibility

## Local Testing Setup
**Running vastai/pytorch image locally for testing:**
- Using `docker run --network host --gpus all` for optimal performance
- Volume mounted to preserve downloaded models between container restarts
- Testing complete setup locally before deploying to vast.ai infrastructure
- Container ID: `b834478b9deb3068e22486b5f92328ef052335828be1ff14d3067e55093d4880`

## Key Findings from vast.ai Template Readme
Based on the template documentation, this Docker image comes with:
- **PyTorch pre-installed** in `/venv/main/` virtual environment (auto-activates)
- **CUDA & cuDNN** already configured
- **uv package manager** for Python packages (`uv pip install`)
- **Jupyter Lab/Notebook** for interactive development
- **Tensorboard** for experiment tracking
- **Node.js & npm** pre-installed
- **Supervisor** for service management
- **Vast.ai CLI** pre-installed
- **Unprivileged Docker container** (some limitations)

## Verified Environment Setup (Local Testing)
✅ **Python 3.10.18** - Confirmed working (matches expected 3.10.x)
✅ **PyTorch 2.4.1+cu124** - Confirmed with CUDA 12.4 support
✅ **CUDA Available** - GPU acceleration working
✅ **Virtual Environment** - `/venv/main/` auto-activates correctly
✅ **GPU Access** - NVIDIA GPUs detected and accessible

## Vast.ai Instance Rental & Setup

### Renting an H100 Instance on Vast.ai

1. **Search for available H100 instances:**
   ```bash
   uv run ./scripts/search_vastai_h100.py
   ```
   This will show available H100 instances sorted by estimated total cost, including minimum bid prices.

2. **Find the cheapest H100 instance:**
   - Look for instances with the lowest "Est$/h" (estimated hourly cost)
   - Check the "Base$/h" and minimum bid requirements
   - Note the instance ID from the first column

3. **Create interruptible instance with minimum bid:**
   ```bash
   vastai create instance <INSTANCE_ID> --image vastai/pytorch:2.4.1-cuda-12.4.1-py310-22.04 --disk 160 --bid
   ```
   - Replace `<INSTANCE_ID>` with the ID from step 2
   - `--bid` creates an interruptible instance (minimum cost)
   - `--disk 160` allocates 160GB container storage
   - The system will use the host's minimum bid price automatically

4. **Wait for instance to become available:**
   - Check status: `vastai show machines`
   - Instance may take a few minutes to start
   - Interruptible instances run when your bid is highest

5. **Get SSH connection details:**
   - **Web Interface (Recommended):** Go to vast.ai dashboard → Instances → Click "SSH" button on your instance card
   - **CLI Alternative:** `vastai show instance <INSTANCE_ID>` for basic info
   - Copy the provided SSH command (includes IP, port, username, and key path)

6. **Connect via SSH:**
   ```bash
   ssh -i /path/to/your/private_key root@IP_ADDRESS -p PORT
   ```
   Use the command provided by the SSH button in the web interface.

### Important Notes for Vast.ai Instances:
- **Interruptible instances** can be paused if someone bids higher
- **Only pay for actual runtime** (billed by the second)
- **Storage costs continue** even when instance is paused
- **Minimum bid** is set by the host - you cannot bid below this
- **160GB disk space** is allocated and cannot be changed later

## Docker Setup Steps

**Important:** All Python/pip commands should be run after the virtual environment activates automatically. If needed manually: `source /venv/main/bin/activate`

### 0. Clone InfiniteTalk Repository
```bash
# Clone the InfiniteTalk repository to get requirements.txt and source code
cd /workspace
git clone https://github.com/MeiGen-AI/InfiniteTalk.git
cd InfiniteTalk
```

### 1. Start Docker Container & Verify Environment
```bash
# Note: Adjust the volume mount path to your local project directory
docker run -it --gpus all \
  -v /path/to/your/project:/workspace \
  vastai/pytorch:2.4.1-cuda-12.4.1-py310-22.04 \
  /bin/bash

# The virtual environment should activate automatically
# Verify Python and PyTorch installation
cd /workspace
python --version  # Should be 3.10.x
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Verify CUDA toolkit
nvcc --version
nvidia-smi

# Check if virtual environment is active
which python
which pip
echo $VIRTUAL_ENV
```

### 2. Install xformers (if not already available)
```bash
# Check if xformers is already installed
python -c "import xformers; print(f'xformers: {xformers.__version__}')" || uv pip install -U xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install Flash Attention
```bash
uv pip install misaki[en]
uv pip install ninja
uv pip install psutil
uv pip install packaging
uv pip install wheel
# Note: flash_attn requires --no-build-isolation to access torch during build
uv pip install --no-build-isolation flash_attn==2.7.4.post1
```

### 4. Install Python Dependencies
```bash
uv pip install -r requirements.txt

# Install librosa (if not already available)
python -c "import librosa; print('librosa available')" || uv pip install librosa

# Additional audio processing dependencies that might be needed
uv pip install soundfile
```

### 5. Install FFmpeg
```bash
apt update && apt install -y ffmpeg
```

### 6. Create Weights Directory Structure
```bash
mkdir -p weights/Wan2.1-I2V-14B-480P
mkdir -p weights/chinese-wav2vec2-base
mkdir -p weights/InfiniteTalk/single
mkdir -p weights/InfiniteTalk/multi
```

### 7. Download Models
```bash
# Install huggingface-cli if not available
python -c "import huggingface_hub; print('huggingface_hub available')" || uv pip install huggingface_hub[cli]

# Download models (this will take significant time and bandwidth)
# Modern approach (recommended):
hf download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./weights/Wan2.1-I2V-14B-480P
hf download TencentGameMate/chinese-wav2vec2-base --local-dir ./weights/chinese-wav2vec2-base
hf download TencentGameMate/chinese-wav2vec2-base model.safetensors --revision refs/pr/1 --local-dir ./weights/chinese-wav2vec2-base

# IMPORTANT: From InfiniteTalk repo, we only need this single file (not the entire repo):
hf download MeiGen-AI/InfiniteTalk single/infinitetalk.safetensors --local-dir ./weights/InfiniteTalk

# Legacy fallback (deprecated but still works):
# huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./weights/Wan2.1-I2V-14B-480P
# huggingface-cli download TencentGameMate/chinese-wav2vec2-base --local-dir ./weights/chinese-wav2vec2-base
# huggingface-cli download TencentGameMate/chinese-wav2vec2-base model.safetensors --revision refs/pr/1 --local-dir ./weights/chinese-wav2vec2-base
# huggingface-cli download MeiGen-AI/InfiniteTalk single/infinitetalk.safetensors --local-dir ./weights/InfiniteTalk
```

### 8. Test Installation
```bash
# Verify all imports work
python -c "
import torch
import torchvision
import torchaudio
from transformers import Wav2Vec2FeatureExtractor
import librosa
import soundfile as sf
print('All core dependencies imported successfully')
"

# Test InfiniteTalk specific imports
python -c "
import sys
sys.path.append('.')
import wan
from src.audio_analysis.wav2vec2 import Wav2Vec2Model
print('InfiniteTalk imports successful')
"
```

### 9. Run Basic Generation Test
```bash
python generate_infinitetalk.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir 'weights/chinese-wav2vec2-base' \
    --infinitetalk_dir weights/InfiniteTalk/single/infinitetalk.safetensors \
    --input_json examples/single_example_image.json \
    --size infinitetalk-480 \
    --sample_steps 40 \
    --mode clip \
    --motion_frame 9 \
    --num_persistent_param_in_dit 0 \
    --save_file test_output
```

## Expected Outcomes After Testing

- [x] Verify Python 3.10.x is available (should be pre-installed)
- [x] Confirm PyTorch 2.4.1 with CUDA 12.4.1 support (should be pre-installed)
- [x] Confirm virtual environment `/venv/main/` auto-activates
- [x] Verify uv package manager is available (should be pre-installed)
- [x] Check if xformers needs installation or is pre-installed (installed xformers==0.0.28 successfully)
- [x] Test flash-attn installation compatibility (installed flash_attn==2.7.4.post1 with --no-build-isolation)
- [x] Verify all pip packages install without conflicts using uv (installed 80 packages from requirements.txt successfully)
- [x] Confirm FFmpeg installation works (pre-installed in vastai/pytorch image, version 4.4.2 - NOTE: quite old but functional)
- [x] Create weights directory structure (created Wan2.1-I2V-14B-480P, chinese-wav2vec2-base, InfiniteTalk directories)
- [ ] Test model downloads (in progress - downloading Wan2.1-I2V-14B-480P model)
- [x] Validate import tests pass (torch, torchvision, torchaudio, transformers, librosa, soundfile, wan, Wav2Vec2Model all imported successfully)
- [x] Validate import tests pass on vast.ai H100 instance (same imports successful on rented instance)
- [ ] Run successful generation test

## Notes for vast.ai Replication

1. **GPU Requirements**: Model requires significant VRAM (14B parameters)
2. **Storage**: ~50GB+ for model weights
3. **Network**: Fast internet for model downloads
4. **Time**: Initial setup may take 1-2 hours

## Testing Checklist

As we execute each step, we'll document:
- What was already available vs what needed installation
- Any errors encountered and their solutions
- Performance metrics (installation time, memory usage)
- Alternative approaches if primary method fails
