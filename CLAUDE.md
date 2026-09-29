# vast_experiments

Provisioning scripts, ComfyUI workflows and benchmarks for running InfiniteTalk / LongCat avatar models on vast.ai GPUs. Knowledge below dates from July 2026; re-check against upstream before relying on it.

## InfiniteTalk fp8 on RTX 5090 (kijai ComfyUI-WanVideoWrapper)
- `WanVideoModelLoader.quantization="disabled"` with an fp8 file stores fp8 but runs matmuls upcast to fp16. The fast `torch._scaled_mm` path only activates when quantization contains `"fast"` (`nodes_model_loading.py`), and it needs the LoRA merged (`fp8_fast` raises NotImplementedError with an unmerged LoRA).
- To enable (Blackwell, cc>=8.9): `quantization=fp8_e4m3fn_fast` AND `WanVideoLoraSelect.merge_loras=True`. Model must be plain (non-scaled) fp8 (`Wan2_1-I2V-14B-480P_fp8_e4m3fn`). Log should say "FP8 matmul enabled"; watch for torch.compile graph-break spam.
- Baseline `workflows/InfiniteTalk-I2V-FP8-Lip-Sync_5090_sage_new_prompts.json` (used by `povision_fp8.sh`) wasted time running 6 steps on a 4-step distill LoRA (lightx2v cfg_step_distill). Candidates: `workflows/IT_{4090,5090}_july2026_{4,5}step.json`. `flowmatch_distill` hard-raises if steps != 4.
- Caching (TeaCache/MagCache/EasyCache) is useless at 4 steps: state resets per window.
- Do NOT switch to Comfy-Org's native `ModelPatchLoader` + `WanInfiniteTalkToVideo` patches: it's kijai's code upstreamed (same math, zero speed gain), fp16 only, no block-swap/TeaCache knobs, and no automatic long-video loop (would need ~20 hand-chained extend nodes for 60 s). Revisit only if Comfy-Org ships an fp8 patch and solves native looping.
- The wrapper is single-GPU only; a faster API provider is likely multi-GPU Ulysses.

## LongCat-Video-Avatar-1.5
- Use the kijai WanVideoWrapper ComfyUI path (example `LongCatAvatar_audio_image_to_video_example_01.json`; fp8 or GGUF + 8-step DMD distill LoRA), not the raw `torchrun` INT8 route in `longcat_video/` (terrible quality on a 5090 per README).
- Models: Kijai HF `Kijai/WanVideo_comfy` + `Kijai/LongCat-Video_comfy` (`LongCat-Avatar-15_bf16` or fp8 scaled, `..._dmd_distill_lora_rank128_bf16`, Wan2_1 VAE bf16, umt5-xxl bf16, chinese-wav2vec2-base, MelBandRoformer). GGUF alt: `vantagewithai/LongCat-Video-Avatar-1.5-GGUF-ComfyUI`.
- Example defaults: 832x480, 93 frames/window, overlap 13, 16 fps, cfg 1.0, 12 steps, `longcat_distill_euler`, sageattn, block_swap 25. On 5090 use 8 steps, block_swap 0. On 4090 (24 GB) keep block_swap ~20.
- Same multitalk/wav2vec embed pipeline as InfiniteTalk. 60 s @480p ~15-24 min on 5090; 720p likely exceeds 30 min.
