#!/usr/bin/env python3
"""Add SDPA fallback to attention module (instead of failing when no flash_attn)."""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/LongCat-Video/longcat_video/modules/avatar/attention.py"

with open(path, 'r') as f:
    content = f.read()

old = '''        else:
            raise RuntimeError("Unsupported attention operations.")'''

new = '''        else:
            # PyTorch native SDPA fallback (works without flash_attn)
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)'''

content = content.replace(old, new)

# Also add the same for the cross-attention path (around line 410+)
old2 = '''        else:
            raise RuntimeError("Unsupported attention operations.")'''
# Replace second occurrence too
content = content.replace(old2, new) if old2 in content else content

with open(path, 'w') as f:
    f.write(content)

count = content.count("scaled_dot_product_attention")
print(f"Patched {path}: {count} SDPA fallback(s) added")
