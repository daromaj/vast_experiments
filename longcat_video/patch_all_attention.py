#!/usr/bin/env python3
"""Add SDPA fallback to ALL attention modules in LongCat-Video."""
import sys, glob

base = sys.argv[1] if len(sys.argv) > 1 else "/workspace/LongCat-Video"

patterns = [
    "longcat_video/modules/attention.py",
    "longcat_video/modules/avatar/attention.py",
]

for pat in patterns:
    path = f"{base}/{pat}"
    with open(path, 'r') as f:
        content = f.read()

    old = '            raise RuntimeError("Unsupported attention operations.")'
    new = '            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)  # SDPA fallback'

    count = content.count(old)
    if count > 0:
        content = content.replace(old, new)
        with open(path, 'w') as f:
            f.write(content)
        print(f"Patched {pat}: {count} fallback(s)")
    else:
        print(f"Skipped {pat}: already patched or not found")
