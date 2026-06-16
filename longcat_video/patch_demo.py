#!/usr/bin/env python3
"""Patch LongCat-Video demo script to fit LongCat-Video Avatar-1.5 in 32 GB VRAM.

Strategy:
- Load text_encoder on CPU (22 GB) — only used once for text encoding
- Wrap DiT loading to force CPU construction, avoid OOM during model init
- Cache-DIT offloads DiT layers to CPU during inference
"""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/LongCat-Video/run_demo_avatar_single_audio_to_video.py"

with open(path, 'r') as f:
    content = f.read()

# 1. Text_encoder stays on CPU
content = content.replace(
    'text_encoder = UMT5EncoderModel.from_pretrained(os.path.join(checkpoint_dir, \'..\', \'LongCat-Video\'), subfolder="text_encoder", torch_dtype=torch.bfloat16)',
    'text_encoder = UMT5EncoderModel.from_pretrained(os.path.join(checkpoint_dir, \'..\', \'LongCat-Video\'), subfolder="text_encoder", torch_dtype=torch.bfloat16, device_map="cpu")'
)

# 2. Exclude text_encoder from pipe constructor
content = content.replace(
    'pipe = LongCatVideoAvatarPipeline(\n        tokenizer = tokenizer,\n        text_encoder = text_encoder,',
    'pipe = LongCatVideoAvatarPipeline(\n        tokenizer = tokenizer,\n        text_encoder = None,  # Re-added after .to(), stays on CPU'
)

# 3. After pipe.to(), re-add text_encoder (still on CPU)
content = content.replace(
    '    pipe.to(local_rank)',
    '    pipe.to(local_rank)\n    pipe.text_encoder = text_encoder  # CPU — only used once for text encoding\n    torch.cuda.empty_cache()\n    print(f"[VRAM] {torch.cuda.memory_allocated()/1024**3:.1f}G used, {torch.cuda.max_memory_allocated()/1024**3:.1f}G peak", flush=True)'
)

# 4. Wrap DiT loading to force CPU construction
old_dit = '''        if use_int8:
            print("[INFO] Loading INT8 quantized DiT model...")
            dit = load_quantized_dit(checkpoint_dir, subfolder="base_model_int8", cp_split_hw=cp_split_hw)'''
new_dit = '''        if use_int8:
            print("[INFO] Loading INT8 quantized DiT model (CPU)...")
            with torch.device("cpu"):
                dit = load_quantized_dit(checkpoint_dir, subfolder="base_model_int8", cp_split_hw=cp_split_hw)
            print("[INFO] DiT loaded on CPU", flush=True)'''
content = content.replace(old_dit, new_dit)

with open(path, 'w') as f:
    f.write(content)

print(f"Patched {path}")
print("  - text_encoder -> CPU, excluded from pipe.to()")
print("  - DiT -> forced CPU construction")
print("  - VRAM print after loading")
