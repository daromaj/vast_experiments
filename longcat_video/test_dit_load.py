#!/usr/bin/env python3
"""Test if DiT model fits in VRAM when loaded in isolation."""
import torch, os, sys
sys.path.insert(0, '/workspace/LongCat-Video')

print(f"CPU free: {os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE') / 1024**3:.1f} GB")
print(f"GPU free: {torch.cuda.mem_get_info()[0]/1024**3:.1f} GB")

from longcat_video.modules.quantization import load_quantized_dit
print("Loading DiT (CPU)...")
dit = load_quantized_dit('/workspace/weights/LongCat-Video-Avatar-1.5', subfolder='base_model_int8')
total_params = sum(p.numel() for p in dit.parameters())
print(f"DiT params: {total_params/1e9:.2f}B")
mem = sum(p.numel() * p.element_size() for p in dit.parameters())
print(f"DiT CPU memory: {mem/1024**3:.1f} GB")

print("Moving DiT to GPU...")
dit = dit.cuda()
print(f"GPU used: {torch.cuda.memory_allocated()/1024**3:.1f} GB")
print(f"GPU free: {torch.cuda.mem_get_info()[0]/1024**3:.1f} GB")
print("SUCCESS - DiT fits on GPU")
