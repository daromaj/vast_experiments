# InfiniteTalk Docker Setup Plan

This document outlines the step-by-step process for setting up InfiniteTalk in a vast ai environment using the `vastai/pytorch:2.4.1-cuda-12.4.1-py310-22.04` image.

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

# Install protobuf (required for CLIP tokenizer)
uv pip install protobuf
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
# LESSON LEARNED: Only InfiniteTalk download can be optimized!
# Download entire Wan2.1-I2V-14B-480P repo (includes all tokenizers and configs)
# Download entire chinese-wav2vec2-base repo (needed for tokenizer/config)
# OPTIMIZE: Only download specific InfiniteTalk model file

# Wan2.1-I2V-14B-480P: Download entire repository (~150GB+)
# This includes all tokenizers, configs, and model files in correct structure
hf download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./weights/Wan2.1-I2V-14B-480P

# chinese-wav2vec2-base: Download entire repository (needed for tokenizer/config)
hf download TencentGameMate/chinese-wav2vec2-base --local-dir ./weights/chinese-wav2vec2-base
hf download TencentGameMate/chinese-wav2vec2-base model.safetensors --revision refs/pr/1 --local-dir ./weights/chinese-wav2vec2-base

# OPTIMIZATION: From InfiniteTalk repo, we only need this single file (not the entire ~24GB repo):
hf download MeiGen-AI/InfiniteTalk single/infinitetalk.safetensors --local-dir ./weights/InfiniteTalk

# Legacy individual file downloads (deprecated - use repo downloads above):
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

### 10. Accelerated Generation Options (Recommended for Speed)

#### Option A: TeaCache + APG (Fastest, ~2-3x speedup)
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
    --use_teacache \
    --teacache_thresh 0.2 \
    --use_apg \
    --apg_momentum -0.75 \
    --apg_norm_threshold 55 \
    --save_file test_output_teacache_apg
```

#### Option B: FusionX LoRA (8 steps instead of 40, ~5x speedup)
```bash
# First download FusionX LoRA (if not available):
hf download vrgamedevgirl84/Wan14BT2VFusioniX FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors --local-dir ./weights

python generate_infinitetalk.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir 'weights/chinese-wav2vec2-base' \
    --infinitetalk_dir weights/InfiniteTalk/single/infinitetalk.safetensors \
    --lora_dir weights/Wan2.1_I2V_14B_FusionX_LoRA.safetensors \
    --lora_scale 1.0 \
    --input_json examples/single_example_image.json \
    --size infinitetalk-480 \
    --sample_steps 8 \
    --sample_text_guide_scale 1.0 \
    --sample_audio_guide_scale 2.0 \
    --sample_shift 2 \
    --mode clip \
    --motion_frame 9 \
    --num_persistent_param_in_dit 0 \
    --save_file test_output_fusionx
```

#### Option C: Lightx2v LoRA (4 steps instead of 40, ~10x speedup)
```bash
# First download Lightx2v LoRA (if not available):
hf download Kijai/WanVideo_comfy Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors --local-dir ./weights

python generate_infinitetalk.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir 'weights/chinese-wav2vec2-base' \
    --infinitetalk_dir weights/InfiniteTalk/single/infinitetalk.safetensors \
    --lora_dir weights/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors \
    --lora_scale 1.0 \
    --input_json examples/single_example_image.json \
    --size infinitetalk-480 \
    --sample_steps 4 \
    --sample_text_guide_scale 1.0 \
    --sample_audio_guide_scale 2.0 \
    --sample_shift 2 \
    --mode clip \
    --motion_frame 9 \
    --num_persistent_param_in_dit 0 \
    --save_file test_output_lightx2v
```

#### Option C+: Lightx2v LoRA + TeaCache + APG (Ultimate Speed, ~12-15x speedup)
```bash
# Combine LoRA distillation with sampling optimizations
python generate_infinitetalk.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir 'weights/chinese-wav2vec2-base' \
    --infinitetalk_dir weights/InfiniteTalk/single/infinitetalk.safetensors \
    --lora_dir weights/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors \
    --lora_scale 1.0 \
    --input_json examples/single_example_image.json \
    --size infinitetalk-480 \
    --sample_steps 4 \
    --sample_text_guide_scale 1.0 \
    --sample_audio_guide_scale 2.0 \
    --sample_shift 2 \
    --mode clip \
    --motion_frame 9 \
    --num_persistent_param_in_dit 0 \
    --use_teacache \
    --teacache_thresh 0.2 \
    --use_apg \
    --apg_momentum -0.75 \
    --apg_norm_threshold 55 \
    --save_file test_output_ultimate_speed
```

#### Option D: Multi-GPU (if you have multiple GPUs)
```bash
# For 2 GPUs (adjust GPU_NUM as needed)
GPU_NUM=2
torchrun --nproc_per_node=$GPU_NUM --standalone generate_infinitetalk.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir 'weights/chinese-wav2vec2-base' \
    --infinitetalk_dir weights/InfiniteTalk/single/infinitetalk.safetensors \
    --dit_fsdp --t5_fsdp \
    --ulysses_size=$GPU_NUM \
    --input_json examples/single_example_image.json \
    --size infinitetalk-480 \
    --sample_steps 40 \
    --mode clip \
    --motion_frame 9 \
    --num_persistent_param_in_dit 0 \
    --save_file test_output_multigpu
```

**IMPORTANT NOTE:** If you encounter the error `ValueError: The output_attentions attribute is not supported when using the attn_implementation set to sdpa`, you need to modify `src/audio_analysis/wav2vec2.py` to force eager attention implementation:

```python
def __init__(self, config: Wav2Vec2Config):
    # Force eager attention implementation to support output_attentions
    config._attn_implementation = "eager"
    super().__init__(config)
```

## Notes for vast.ai Replication

1. **GPU Requirements**: Model requires significant VRAM (14B parameters)
2. **Storage**: ~160GB+ for model weights
3. **Network**: Fast internet for model downloads
4. **Time**: Initial setup may take 1-2 hours

## Performance Benchmarks (10-second audio, 480p)

### Baseline Performance:
- **Default 40-step generation**: ~28 minutes 31 seconds total
  - Model loading: ~2 minutes
  - Generation: ~26 minutes (40 steps × ~39s/step)
  - Video: 3.24s, 424.8 kbps, 166KB

### Optimized Performance:
- **Lightx2v LoRA (4 steps)**: **3 minutes 46 seconds** (~10x speedup)
  - Model loading: ~2 minutes
  - Generation: ~1 minute 46 seconds (4 steps × ~26s/step)
  - Video: 3.24s, 461.3 kbps, 184KB

- **TeaCache + APG (40 steps)**: ~8-12 minutes estimated (2-3x speedup)
- **FusionX LoRA (8 steps)**: ~5-7 minutes estimated (5x speedup)

## Testing Checklist

As we execute each step, we'll document:
- What was already available vs what needed installation
- Any errors encountered and their solutions
- Performance metrics (installation time, memory usage)
- Alternative approaches if primary method fails
