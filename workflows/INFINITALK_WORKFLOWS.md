# InfiniTalk ComfyUI Workflows

## File Inventory

### Modern (June 2026) — **use these**

| File | GPU | Format | BlockSwap |
|---|---|---|---|
| `InfiniteTalk-I2V-FP8-Lip-Sync_5090_modern.json` | RTX 5090 (32GB) | UI (drag & drop) | 5 |
| `InfiniteTalk-I2V-FP8-Lip-Sync_4090_modern.json` | RTX 4090 (24GB) | UI (drag & drop) | 15 |
| `InfiniteTalk-I2V-FP8-Lip-Sync_5090_modern_API.json` | RTX 5090 (32GB) | API (programmatic) | 5 |
| `InfiniteTalk-I2V-FP8-Lip-Sync_4090_modern_API.json` | RTX 4090 (24GB) | API (programmatic) | 15 |

### Legacy (Dec 2025) — kept for reference

| File | Notes |
|---|---|
| `InfiniteTalk-I2V-FP8-Lip-Sync_5090_sage_new_prompts.json` | Previous 5090 workflow |
| `InfiniteTalk-I2V-FP8-Lip-Sync_4090_sage_new_prompts.json` | Previous 4090 workflow |
| `InfiniteTalk-I2V-FP8-Lip-Sync_5090_sage_new_prompts_API.json` | Previous 5090 API workflow |
| `InfiniteTalk-I2V-FP8-Lip-Sync_4090_sage_new_prompts_API.json` | Previous 4090 API workflow |
| `../InfiniteTalk-I2V-FP8-Lip-Sync.json` | Root-level base workflow (81KB) |

### Reference

| File | Notes |
|---|---|
| `InfiniteTalk-I2V-FP8-Lip-Sync_official_reference_2025-08.json` | Official MeiGen-AI example (Aug 2025, unmodified) |

## What Changed (Dec 2025 → June 2026)

| Setting | Old | New | Why |
|---|---|---|---|
| Steps | 6 | 4 | `flowmatch_distill` is tuned for 4-step inference |
| Sampler | `dpm++_sde` | `flowmatch_distill` | Faster convergence, same perceptual quality |
| LoRA weight | 1.0 | 0.8 | Less overfitting, better generalization across faces |
| BlockSwap | 0 | 5 (5090) / 15 (4090) | Frees VRAM for context windows, zero speed penalty |
| AudioCrop | absent | added | Trims silence from input audio (from official workflow) |
| ContextOptions | absent | `uniform_looped`, 81f, stride 4, overlap 16 | Windowed context blending — better consistency on videos >81 frames |
| EnhanceAVideo | absent | weight 2.0, full range | FETA-based quality enhancement during decode |
| LoopArgs | absent | shift_skip=6, full range | Streaming loop latent shift for long generations |
| TorchCompileSettings | already present | unchanged | `inductor` backend, compile transformer blocks only |
| Note node | present | removed | Was just a Triton reminder, TorchCompile is now wired |

## Required Custom Nodes

All from ComfyUI Manager or manual install:

| Node | Repo | Notes |
|---|---|---|
| **ComfyUI-WanVideoWrapper** | `kijai/ComfyUI-WanVideoWrapper` | **Critical.** Must be on version ≥ May 2026 (MultiTalk fixes). Our workflows target commit `c3ee35f3` baseline, but newer is fine. |
| **ComfyUI-VideoHelperSuite** | `Kosinkadink/ComfyUI-VideoHelperSuite` | VHS_VideoCombine for output |
| **ComfyUI-MelBandRoFormer** | `kijai/ComfyUI-MelBandRoFormer` | Audio preprocessing for cleaner voice extraction |
| **ComfyUI-KJNodes** | `kijai/ComfyUI-KJNodes` | ImageResizeKJv2, GetImageSizeAndCount |

Install all at once:

```bash
cd /workspace/ComfyUI/custom_nodes
git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
git clone https://github.com/kijai/ComfyUI-MelBandRoFormer.git
git clone https://github.com/kijai/ComfyUI-KJNodes.git
```

## Required Models

### Diffusion Models (`models/diffusion_models/`)

| File | Size | Purpose |
|---|---|---|
| `Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors` | ~15.8 GB | Main I2V base model |
| `Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors` | ~2.5 GB | InfiniTalk single-person adapter |
| `MelBandRoformer_fp16.safetensors` | ~435 MB | Audio feature extraction |

### Text Encoders (`models/text_encoders/`)

| File | Size | Purpose |
|---|---|---|
| `umt5-xxl-enc-bf16.safetensors` | ~10.6 GB | Prompt encoding |

### CLIP Vision (`models/clip_vision/`)

| File | Size | Purpose |
|---|---|---|
| `clip_vision_h.safetensors` | ~1.2 GB | Reference image identity encoding |

### VAE (`models/vae/`)

| File | Size | Purpose |
|---|---|---|
| `Wan2_1_VAE_bf16.safetensors` | ~242 MB | Latent-to-RGB decoding |

### LoRAs (`models/loras/`)

| File | Size | Purpose |
|---|---|---|
| `lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors` | ~703 MB | 4-step distillation + CFG distillation |

**Total: ~31 GB** (FP8 models). Wav2Vec2 (`chinese-wav2vec2-base`) downloads automatically on first run (~500 MB).

### Model Download (manual)

```bash
# Diffusion models
wget -P models/diffusion_models/ \
  "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors" \
  "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors" \
  "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/MelBandRoformer_fp16.safetensors"

# Text encoder
wget -P models/text_encoders/ \
  "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-bf16.safetensors"

# CLIP vision
wget -P models/clip_vision/ \
  "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/clip_vision_h.safetensors"

# VAE
wget -P models/vae/ \
  "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors"

# LoRA
wget -P models/loras/ \
  "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"
```

Or use the provisioning script: `../povision_fp8.sh` (handles all of the above + SageAttention compilation).

## GPU-Specific Configuration

| Parameter | RTX 4090 (24GB) | RTX 5090 (32GB) |
|---|---|---|
| BlockSwap blocks | 15 | 5 |
| Resolution | 832×480 | 832×480 |
| Frame window | 81 | 81 |
| Context stride | 4 | 4 |
| Context overlap | 16 | 16 |
| Steps | 4 | 4 |
| Sampler | flowmatch_distill | flowmatch_distill |
| LoRA weight | 0.8 | 0.8 |
| CFG | 1.0 | 1.0 |
| Shift | 11.0 | 11.0 |

**4090 note:** BlockSwap=15 means more model layers are offloaded to CPU RAM during inference. This frees ~2-3GB VRAM with negligible speed impact (<2%). If you have a PCIe 4.0 x16 slot, the offload overhead is effectively zero.

**5090 note:** BlockSwap=5 is conservative. You can try 0 if you want all blocks in VRAM — the 32GB buffer should handle it for the FP8 models. If you hit OOM, bump to 5-10.

## Parameter Reference

### WanVideoSampler

| Param | Value | Notes |
|---|---|---|
| `steps` | 4 | Tuned for flowmatch_distill + lightx2v LoRA |
| `cfg` | 1.0 | CFG disabled (distilled model doesn't need it) |
| `shift` | 11.0 | Timestep shift for Wan 2.1 schedules |
| `scheduler` | `flowmatch_distill` | Must match lightx2v LoRA training |
| `denoise` | 1.0 | Full denoise for I2V |
| `seed` | randomized | Set to fixed for reproducible outputs |
| `force_offload` | true | Offloads model between windows |

### WanVideoContextOptions

| Param | Value | Notes |
|---|---|---|
| `context_schedule` | `uniform_looped` | Best for talking head consistency |
| `context_frames` | 81 | ~3.2s at 25fps per window |
| `context_stride` | 4 | Latent stride (4 pixel frames = 1 latent frame) |
| `context_overlap` | 16 | Overlap between windows for blending |
| `freenoise` | true | Shuffles noise per window (better diversity) |
| `fuse_method` | `linear` | Linear crossfade at window edges |

### WanVideoImageToVideoMultiTalk

| Param | Value | Notes |
|---|---|---|
| `width` / `height` | 832 / 480 | Portrait 9:16 (swap for landscape) |
| `frame_window_size` | 81 | Must match context_frames |
| `motion_frame` | 9 | Reference frames for motion extraction |
| `mode` | `infinitetalk` | Required for InfiniTalk model |
| `colormatch` | `disabled` | Enable if source image has color cast |
| `force_offload` | false | Keep MultiTalk model in VRAM |
| `tiled_vae` | false | Enable if OOM during VAE decode |

### MultiTalkWav2VecEmbeds

| Param | Value | Notes |
|---|---|---|
| `normalize_loudness` | true | Normalizes audio input level |
| `num_frames` | 2000 | Max frames to process (80s at 25fps) |
| `fps` | 25 | Must match VHS_VideoCombine frame_rate |
| `audio_scale` | 1 | Audio influence strength |
| `audio_cfg_scale` | 1 | Audio CFG guidance |

## Performance Benchmarks

Measured on vast.ai instances with SageAttention 2:

| GPU | Audio Duration | Time | Windows | Per-Window |
|---|---|---|---|---|
| RTX 5090 | 5s | <102s | 2 | ~51s |
| RTX 5090 | 58s | ~12m | 21 | ~34s |
| RTX 4090 | 58s | ~19m | 21 | ~46s |
| RTX 4090 | 10s | ~238s | 4 | ~60s |

**Setup time:** ~8 min (ComfyUI startup + SageAttention build on vast.ai).  
**Cost estimate:** ~$0.30 per 60s video on a $0.60/hr instance.

## Common Issues

### "Cannot find model Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors"
Model file missing or wrong path. Verify it's in `models/diffusion_models/`. Case-sensitive.

### OOM during sampling
1. Increase BlockSwap (15→20, or 5→15)
2. Enable `force_offload` on sampler (should already be on)
3. Enable `tiled_vae` on MultiTalk node
4. Reduce context_frames from 81 to 61 or 41 (shorter windows, less VRAM per pass)

### Color cast or washed-out output
The `colormatch` param on MultiTalk node can help. Try `"mkl"` or `"hm"` modes. Default is `"disabled"` — enable if you see color shifts.

### Long video tail drops frames
This was a known bug in pre-2026 versions, fixed in the v3.0-era ContextOptions approach. The `context_overlap=16` ensures proper window blending. If you still see it, increase overlap to 32.

### "MultiTalkWav2VecEmbeds: audio stride mismatch"
This is the **May 2026 fix** (`Fix multitalk_audio_stride init`). Update ComfyUI-WanVideoWrapper to latest commit.

### FlowMatch sampler gives worse results than dpm++_sde
The lightx2v LoRA is specifically trained for flowmatch_distill. If you swap to a non-distilled LoRA (like regular lightx2v), switch sampler back to `dpm++_sde` and increase steps to 6-8.

### SageAttention not working
1. Check CUDA version: `python -c "import torch; print(torch.version.cuda)"`
2. For 4090: needs `TORCH_CUDA_ARCH_LIST="8.9"` during build
3. For 5090: needs `TORCH_CUDA_ARCH_LIST="12.0"` during build
4. Fall back to `flash_attn` if SageAttention build fails — ~15-20% slower but works everywhere

### ComfyUI loads workflow but nodes are red (missing)
Use ComfyUI Manager → "Install Missing Custom Nodes." The four required node packs are listed above. If Manager can't find them, install manually from GitHub.

### Audio sounds distorted or lip sync is off
1. Check `normalize_loudness: true` on MultiTalkWav2VecEmbeds
2. Verify AudioCrop isn't cutting into the speech (default 0:00–10:00 is generous)
3. MelBandRoFormer preprocessing can sometimes over-denoise — try bypassing it by connecting LoadAudio directly to MultiTalkWav2VecEmbeds (removing audio clarity but eliminating a potential failure point)

### Workflow got corrupted during JSON editing
All originals are preserved. Restore from the `*_sage_new_prompts*.json` files and re-apply modifications manually.

## Workflow Architecture

```
LoadImage → ImageResizeKJv2 → GetImageSizeAndCount ──→ WanVideoClipVisionEncode ──→ WanVideoImageToVideoMultiTalk ──→ WanVideoSampler
                                    │                        ↑                                  ↑                              │
                                    └── width/height ────────┘                                  │                              │
                                                                                               │                              │
LoadAudio → AudioCrop ──→ MelBandRoFormerModelLoader → MelBandRoFormerSampler → MultiTalkWav2VecEmbeds ──────────────────────┘
                │                                                                                                                  │
                └──────────────────────────────────────────────────────────────────→ VHS_VideoCombine ←──────────────────────────┘
                                                                                                ↑
WanVideoModelLoader ←── CLIPVisionLoader                                                       │
       ↑                                                                                       │
       ├── WanVideoLoraSelect                                                                   │
       ├── WanVideoBlockSwap                                                                    │
       ├── MultiTalkModelLoader                                                                 │
       └── TorchCompileSettings                                                                 │
                                                                                                │
WanVideoVAELoader ─────────────────────────────────────────────────────────────────────────────┘

New nodes (modern only):
  WanVideoContextOptions ──→ WanVideoSampler.context_options
  WanVideoEnhanceAVideo  ──→ WanVideoSampler.feta_args
  WanVideoLoopArgs       ──→ WanVideoSampler.loop_args
```

## Model Version Notes

The official `MeiGen-AI/InfiniteTalk` GitHub repository's `comfyui` branch has been frozen since **August 2025**. The HuggingFace model weights (`MeiGen-AI/InfiniteTalk`) last updated **September 2025**. The ComfyUI-WanVideoWrapper custom node package (which contains the actual MultiTalk/InfiniTalk implementation) is actively maintained by `kijai` — last updated **May 2026**.

This means:
- **Model weights are stable** — no new InfiniTalk model versions since Sep 2025
- **Node code is active** — WanVideoWrapper gets regular fixes and features
- **Workflows are community-driven** — the official examples are stale, community workflows (like ours) are the current best practice

When troubleshooting, always check the WanVideoWrapper commit you're running against the latest release. The node pack is where bugs get fixed.
