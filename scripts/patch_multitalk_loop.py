#!/usr/bin/env python3
"""
Patch ComfyUI-WanVideoWrapper's multitalk_loop.py with two speed fixes that have
no effect on output. Run this ON the instance.

Both are gated behind environment variables so one push can A/B every
combination without re-patching:

    WANOPT_Y_CACHE=1          enable the redundant-encode cache (R3)
    WANOPT_KEEP_CACHE_WARM=1  skip the per-window empty_cache/gc (R6)

Unset (the default) reproduces stock behaviour exactly, so a regression can
always be bisected without restoring the file.

R3 - the 81-frame `y` VAE encode is recomputed identically every window.
  cond_ is `cond_image if (is_first_clip or humo_image_cond is None) else
  cond_frame`; humo_image_cond is None in this workflow, so cond_ is ALWAYS
  cond_image. cond_image is original_images[:, :, i:i+1] and original_images is
  a single start image whose last frame is simply repeated when the index runs
  past the end. The other 80 frames of padding_frames_pixels_values are zeros.
  vae.encode is deterministic (sample=False returns mu), so the result is
  bit-identical across all N windows while costing 21 encoder iterations each
  time - roughly half the per-window VAE work.

  The cache is keyed on a content fingerprint of the actual input tensor, not on
  an assumption about the workflow, so an image-sequence-driven run (where
  cond_ genuinely changes) simply misses and recomputes. Correctness does not
  depend on the reasoning above being right.

R6 - mm.soft_empty_cache() + gc.collect() run every window. empty_cache()
  returns the caching allocator's blocks to the driver and synchronises, so the
  next window re-cudaMallocs its whole working set. These exist to hold peak
  VRAM down; at 24.9 of 31.36 GiB there is headroom to skip them. This is the
  first thing to put back if a config raises peak memory.
"""
import argparse
import os
import shutil

TARGET = ("/workspace/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper.git"
          "/multitalk/multitalk_loop.py")

# --- R3 -----------------------------------------------------------------
Y_OLD = """            # encode
            vae.to(device)
            y = vae.encode(padding_frames_pixels_values, device=device, tiled=tiled_vae, pbar=False).to(dtype)[0]
"""

Y_NEW = '''            # encode
            vae.to(device)
            if os.environ.get("WANOPT_Y_CACHE") == "1":
                # Fingerprint the real tensor rather than trusting that the
                # workflow keeps cond_ constant: a run driven by an image
                # sequence must miss and recompute.
                _fp = (
                    tuple(padding_frames_pixels_values.shape),
                    str(padding_frames_pixels_values.dtype),
                    float(padding_frames_pixels_values.sum()),
                    float(padding_frames_pixels_values[:, :, :1].abs().mean()),
                    bool(tiled_vae),
                )
                _hit = globals().get("_WANOPT_Y_FP") == _fp
                if _hit:
                    y = globals()["_WANOPT_Y_VAL"]
                else:
                    y = vae.encode(padding_frames_pixels_values, device=device, tiled=tiled_vae, pbar=False).to(dtype)[0]
                    globals()["_WANOPT_Y_FP"] = _fp
                    globals()["_WANOPT_Y_VAL"] = y
                log.info(f"[WANOPT] y-encode cache {'HIT' if _hit else 'MISS'}")
            else:
                y = vae.encode(padding_frames_pixels_values, device=device, tiled=tiled_vae, pbar=False).to(dtype)[0]
'''

# --- R6 -----------------------------------------------------------------
# Two separate sites; anchor each on its unique preceding line so the 12-space
# and 8-space variants cannot be confused.
CACHE1_OLD = """            y = torch.cat([msk, y]) # 4+C T H W
            mm.soft_empty_cache()
"""

CACHE1_NEW = """            y = torch.cat([msk, y]) # 4+C T H W
            if os.environ.get("WANOPT_KEEP_CACHE_WARM") != "1":
                mm.soft_empty_cache()
"""

CACHE2_OLD = """        mm.soft_empty_cache()
        gc.collect()
        # sampling loop
"""

CACHE2_NEW = """        if os.environ.get("WANOPT_KEEP_CACHE_WARM") != "1":
            mm.soft_empty_cache()
            gc.collect()
        # sampling loop
"""

PATCHES = [("R3 y-encode cache", Y_OLD, Y_NEW),
           ("R6 empty_cache site 1", CACHE1_OLD, CACHE1_NEW),
           ("R6 empty_cache site 2", CACHE2_OLD, CACHE2_NEW)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", default=TARGET)
    ap.add_argument("--restore", action="store_true",
                    help="put the .orig backup back and exit")
    args = ap.parse_args()

    backup = args.target + ".orig"

    if args.restore:
        if not os.path.exists(backup):
            print(f"no backup at {backup}")
            return 1
        shutil.copy2(backup, args.target)
        print(f"restored {args.target} from backup")
        return 0

    src = open(args.target).read()

    if "WANOPT_Y_CACHE" in src:
        print("already patched - nothing to do")
        return 0

    if not os.path.exists(backup):
        shutil.copy2(args.target, backup)
        print(f"backup written to {backup}")

    for name, old, new in PATCHES:
        if old not in src:
            print(f"FATAL: anchor not found for {name} - refusing to patch")
            return 1
        if src.count(old) != 1:
            print(f"FATAL: anchor for {name} matches {src.count(old)}x - ambiguous")
            return 1
        src = src.replace(old, new)
        print(f"  applied {name}")

    # The module already imports os at line 2, so the env lookups resolve.
    if "\nimport os\n" not in src:
        print("FATAL: module does not import os")
        return 1

    open(args.target, "w").write(src)
    print(f"patched {args.target}")

    compile(src, args.target, "exec")
    print("syntax OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
