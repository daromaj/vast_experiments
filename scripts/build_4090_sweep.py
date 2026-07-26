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

MEASURED 2026-07-27, first pass: q0 OOMed in WanVideoSampler after 407.5s and q1
OOMed too, so tiled VAE alone does not close the gap either. The failures also
reported the real ceiling: torch says "Device limit: 23.52 GiB", not the 23.99GiB
the 24,564MB nameplate suggests - roughly 480MB goes to driver and context. The
gap versus the 5090's ~24.4GiB peak is therefore ~0.9GiB of *usable* memory, and
some block swapping is mandatory on this card.

Second pass drops swap=0 as settled and brackets the minimum from below (2, 4)
rather than starting at the 6 the arithmetic predicted, since the true floor is
what determines the fastest config.
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
    ("q0_swap00_untiled", 0, False),   # MEASURED: OOM at 407.5s in WanVideoSampler
    ("q1_swap00_tiled", 0, True),      # MEASURED: OOM - tiled VAE alone is not enough
    ("q2_swap06_untiled", 6, False),   # what the per-block arithmetic predicts
    ("q3_swap06_tiled", 6, True),      # belt and braces
    ("q4_swap12_untiled", 12, False),  # halfway to the shipped default
    ("q5_swap20_tiled", 20, True),     # the shipped default, as the control
]

# Second pass. ~0.9GiB to shed at ~443MB per block is 2-3 blocks, so the floor is
# probably below the 6 that pass one started from; every block above the floor is
# CPU->GPU traffic on every forward, paid for nothing.
CANDIDATES_PASS2 = [
    ("q6_swap02_untiled", 2, False),   # bare minimum the arithmetic allows
    ("q7_swap04_untiled", 4, False),   # one block of margin
    ("q2_swap06_untiled", 6, False),
    ("q4_swap12_untiled", 12, False),
    ("q3_swap06_tiled", 6, True),
    ("q5_swap20_tiled", 20, True),
]


# Third pass, and the result that matters. Pass two refuted the premise of the
# whole sweep: swap=4 and swap=6 both ran at ~255s while swap=12 ran at 108.9s.
# More swapping is FASTER here, 2.4x so.
#
# The peak-VRAM column explains it. swap=4 and swap=6 both sat at ~24,076MB,
# which is the 23.52GiB device limit exactly; swap=12 peaked at 21,782MB and was
# the first variant with actual headroom. Pinned against the ceiling the caching
# allocator thrashes - synchronous frees and cudaMalloc retries - and that costs
# far more than moving 8 extra blocks across PCIe. "Just barely fits" is the
# worst operating point on this card, not the best.
#
# So the question is no longer "what is the minimum that fits" but "where does
# the headroom benefit stop paying for the transfer cost", which is above 12.
CANDIDATES_PASS3 = [
    ("q8_swap16_untiled", 16, False),
    ("q9_swap20_untiled", 20, False),  # shipped default's swap count, untiled
    ("qa_swap28_untiled", 28, False),  # past the point transfers should dominate
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
    ap.add_argument("--pass2", action="store_true",
                    help="emit the post-OOM candidate set instead of the first pass")
    ap.add_argument("--pass3", action="store_true",
                    help="emit the high-swap set, after more swap proved faster")
    args = ap.parse_args()

    base = json.load(open(BASE))
    os.makedirs(args.outdir, exist_ok=True)

    sets = {True: CANDIDATES_PASS3} if args.pass3 else {True: CANDIDATES_PASS2}
    chosen = sets[True] if (args.pass2 or args.pass3) else CANDIDATES

    for name, swap, tiled in chosen:
        wf = build(base, swap, tiled, args.image, args.audio)
        path = os.path.join(args.outdir, f"{name}_API.json")
        with open(path, "w") as fh:
            json.dump(wf, fh, indent=1)
        print(f"{name:<20} swap={swap:<3} prefetch={1 if swap else 0} "
              f"tiled_vae={tiled} -> {path}")


if __name__ == "__main__":
    raise SystemExit(main())
