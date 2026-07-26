#!/usr/bin/env python3
"""
Emit a block-swap sweep for a 24GB RTX 4090, derived from the tuned 5090 config.

Why swap is needed at all: the 480p pipeline peaked at ~24.4GiB on a 5090 (polled
nvidia-smi reserved, so a close lower bound). A 4090 has 24,564MiB ~= 23.99GiB
total and perhaps 23.5GiB usable, so it is roughly 1GiB short - "almost fits".

How much to shed: the run log reports 17,718MB of transformer blocks resident
across 40 blocks, i.e. ~443MB per block. Covering a ~1.4GiB gap needs about
1400/443 ~= 3.2 blocks, so 4-6 with margin. The shipped 4090 workflows carry
blocks_to_swap=20, which is the NODE DEFAULT and sheds ~8.9GiB - roughly 6x more
than the arithmetic calls for, and every swapped block is CPU->GPU traffic on each
forward. This sweep is here to replace that guess with a measurement.

prefetch_blocks=1 whenever swapping, to overlap the transfer with compute
(model.py:3243 only reads it when blocks_to_swap > 0).

tiled_vae is the other headroom lever and is cheaper than it looks here: it cost
23s on a 5090, but on a card that would otherwise have to swap more blocks it can
come out ahead.

q0 is expected to OOM. It is included deliberately - it establishes the ceiling
rather than assuming it, and it is the only run that tells us whether the ~1GiB
estimate was right.
"""
import argparse
import copy
import json
import os

BASE = "workflows/generated/sweep/s8_sage_quickwins_API.json"

N_SWAP = "134"
N_I2V = "192"
N_LOAD_IMAGE = "284"
N_LOAD_AUDIO = "125"

# name, blocks_to_swap, tiled_vae
CANDIDATES = [
    ("q0_swap00_untiled", 0, False),   # does it actually fit? expected OOM
    ("q1_swap00_tiled", 0, True),      # tiled VAE alone might close a ~1GiB gap
    ("q2_swap06_untiled", 6, False),   # what the per-block arithmetic predicts
    ("q3_swap06_tiled", 6, True),      # belt and braces
    ("q4_swap12_untiled", 12, False),  # halfway to the shipped default
    ("q5_swap20_tiled", 20, True),     # the shipped default, as the control
]


def build(base, swap, tiled, image, audio):
    wf = copy.deepcopy(base)
    s = wf[N_SWAP]["inputs"]
    s["blocks_to_swap"] = swap
    # Only meaningful when swapping; overlaps the CPU->GPU copy with compute.
    s["prefetch_blocks"] = 1 if swap > 0 else 0
    wf[N_I2V]["inputs"]["tiled_vae"] = tiled
    wf[N_LOAD_IMAGE]["inputs"]["image"] = image
    wf[N_LOAD_AUDIO]["inputs"]["audio"] = audio
    return wf


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", default="santa-classic-portrait.png")
    ap.add_argument("--audio", default="santa_8s.mp3")
    ap.add_argument("--outdir", default="workflows/generated/sweep4090")
    args = ap.parse_args()

    base = json.load(open(BASE))
    os.makedirs(args.outdir, exist_ok=True)

    for name, swap, tiled in CANDIDATES:
        wf = build(base, swap, tiled, args.image, args.audio)
        path = os.path.join(args.outdir, f"{name}_API.json")
        with open(path, "w") as fh:
            json.dump(wf, fh, indent=1)
        print(f"{name:<20} swap={swap:<3} prefetch={1 if swap else 0} "
              f"tiled_vae={tiled} -> {path}")


if __name__ == "__main__":
    raise SystemExit(main())
