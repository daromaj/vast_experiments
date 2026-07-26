#!/usr/bin/env python3
"""
Compare SageAttention against SDPA at the exact shape WanVideo uses - run ON the
instance.

Context: attention_mode=sageattn OOMs these workflows in WanVideoSampler at
29.3GiB while sdpa completes at 23.4GiB peak. A fused kernel using MORE memory
than SDPA is backwards, and sage_abi_probe already passes (cosine 0.9993), so
the build is fine and the problem is shape-specific. This measures peak memory
and latency per backend at the real shape so the choice is made on numbers.

Shape: an 81-frame 480x832 window is 21 latent frames (temporal compression 4)
of 30x52 patches (patch size 1,2,2) = 32,760 tokens. Wan 14B is 40 heads x
head_dim 128.

    python3 attention_bench.py            # real shape
    python3 attention_bench.py --json      # machine readable
"""
import argparse
import json
import time

import torch
import torch.nn.functional as F

SEQ = 32760
HEADS = 40
HEAD_DIM = 128


def measure(fn, q, k, v, warmup=1, iters=3):
    """Peak VRAM and median latency for one backend, or the error that killed it."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        for _ in range(warmup):
            fn(q, k, v)
        torch.cuda.synchronize()

        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            out = fn(q, k, v)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        times.sort()
        return {
            "ok": True,
            "median_s": times[len(times) // 2],
            "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
            "has_nan": bool(torch.isnan(out).any().item()),
        }
    except Exception as e:  # noqa: BLE001 - an OOM here is a result, not a crash
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq", type=int, default=SEQ)
    ap.add_argument("--heads", type=int, default=HEADS)
    ap.add_argument("--head-dim", type=int, default=HEAD_DIM)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    report = {
        "shape": {"seq": args.seq, "heads": args.heads, "head_dim": args.head_dim},
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    }

    q, k, v = (
        torch.randn(1, args.heads, args.seq, args.head_dim,
                    device="cuda", dtype=torch.float16)
        for _ in range(3)
    )
    qkv_gib = 3 * q.nelement() * q.element_size() / 2**30
    report["qkv_gib"] = qkv_gib

    backends = {"sdpa": lambda a, b, c: F.scaled_dot_product_attention(a, b, c)}
    try:
        from sageattention import sageattn

        backends["sageattn"] = lambda a, b, c: sageattn(a, b, c, tensor_layout="HND")
    except Exception as e:  # noqa: BLE001
        report["sageattn_import_error"] = repr(e)

    report["results"] = {name: measure(fn, q, k, v) for name, fn in backends.items()}

    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return 0

    print(json.dumps(report, indent=2, default=str))
    print("\n" + "=" * 62)
    print(f"shape: {args.heads}h x {args.seq} tok x {args.head_dim}d  "
          f"(Q+K+V = {qkv_gib:.2f} GiB)")
    for name, r in report["results"].items():
        if r.get("ok"):
            print(f"  {name:<10} {r['median_s'] * 1000:8.1f} ms   "
                  f"peak {r['peak_gib']:6.2f} GiB   nan={r['has_nan']}")
        else:
            print(f"  {name:<10} FAILED: {r['error']}")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
