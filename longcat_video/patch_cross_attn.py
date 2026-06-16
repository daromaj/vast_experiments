#!/usr/bin/env python3
"""Fix SDPA fallback for cross-attention in modules/attention.py."""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/LongCat-Video/longcat_video/modules/attention.py"

with open(path, 'r') as f:
    content = f.read()

# Replace the SDPA fallback in _process_cross_attn (cross-attention needs special handling)
old = '''        else:
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)  # SDPA fallback


        x = x.view(B, -1, C)'''

new = '''        else:
            # SDPA: need [B, H, S, D] format; q/k/v are in [1, total_seq, H, D]
            q_sdpa = q.permute(0, 2, 1, 3)  # [1, heads, seq_q, dim]
            k_sdpa = k.permute(0, 2, 1, 3)
            v_sdpa = v.permute(0, 2, 1, 3)
            x = torch.nn.functional.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa)
            x = x.permute(0, 2, 1, 3)  # back to [1, seq, heads, dim]


        x = x.view(B, -1, C)'''

content = content.replace(old, new)

# Also fix _process_attn in avatar/attention.py (self-attention — simpler)
path2 = path.replace("modules/attention.py", "modules/avatar/attention.py")
with open(path2, 'r') as f:
    content2 = f.read()

old2 = '''        else:
            # PyTorch native SDPA fallback (works without flash_attn)
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)'''

new2 = '''        else:
            # SDPA: q,k,v in [B, H, S, D] format (from rearrange or default)
            if q.ndim == 4 and q.shape[1] != k.shape[1]:
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)
                x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                x = x.transpose(1, 2)
            else:
                x = torch.nn.functional.scaled_dot_product_attention(q, k, v)'''

if old2 in content2:
    content2 = content2.replace(old2, new2)
    with open(path2, 'w') as f:
        f.write(content2)
    print(f"Patched {path2}")
else:
    print(f"Skipped {path2}: pattern not found")

with open(path, 'w') as f:
    f.write(content)

print(f"Patched {path}: cross-attention SDPA with transpose")
