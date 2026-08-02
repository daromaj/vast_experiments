#!/usr/bin/env python3
"""
What does bandwidth pricing actually cost us per video, and where is the cliff?

    scripts/egress_cost.py                 # cost table from the known payload
    scripts/egress_cost.py --market        # + what the live market offers per tier

Vast prices bandwidth per direction, and the web UI shows the pair stacked
(up over down) while the API calls them inet_up_cost / inet_down_cost. They
invert relative to each other, which is a good way to cap the wrong one. The
line items settle it with no interpretation needed - the quantities give it
away:

    "download charge: GB * $/GB"   quantity 34.600   rate 0.003   $0.090
    "upload charge:   GB * $/GB"   quantity  0.189   rate 0.004   $0.001

34.6 GB is the model payload arriving. 0.189 GB is the finished video leaving.
So DOWNLOAD is the one that matters, by a factor of ~180x in volume.

Units: vast bills in binary GB and converts at 1024, not 1000. Confirmed from
the raw offer fields, which agree exactly:
    inet_down_cost 0.0026041666 $/GB x 1024 = 2.6667 = internet_down_cost_per_tb
    inet_up_cost   0.00390625   $/GB x 1024 = 4.0000 = internet_up_cost_per_tb
Using 1000 here would understate every figure below by 2.4%.
"""
import argparse
import json
import subprocess
import sys

GIB = 1024  # vast's GB per TB

# The model payload, from povision_fp8.sh. Sizes as that file documents them.
MODELS = [
    ("Wan2_1-I2V-14B-480P_fp8_e4m3fn", 15.83),
    ("umt5-xxl-enc-bf16", 10.58),
    ("Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ", 2.53),
    ("clip_vision_h", 1.18),
    ("lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16", 0.703),
    ("MelBandRoformer_fp16", 0.435),
    ("Wan2_1_VAE_bf16", 0.242),
    ("wav2vec2-chinese-base_fp16", 0.190),
]
# povision_fp8.sh TOTAL_BYTES_TO_DOWNLOAD, the authoritative figure.
MODELS_GIB = 33978384650 / 1024**3

# Still pulled on every rental, in the background since 8547f71. Deferring it
# into the fallback branch would remove it from the bill as well as the clock.
CUDA_DEV_GIB = 2093 / 1024  # measured: "Fetched 2093 MB"

# Measured, not modelled: what vast actually billed for one full one-shot
# rental (instance 46593521). The gap over models+cuda is pip, the wheel, hf
# metadata and apt lists - real traffic we do not itemise.
BILLED_GIB = 34.6

# The finished video going home. Two 58 s mp4s (silent + audio) per render.
UPLOAD_GIB = 0.189

# The container image is pulled before provisioning starts. It costs TIME but
# not MONEY - the 34.6 GB billed above is the models alone, with no sign of the
# 9.5 GB image in it.
IMAGE_GIB = 9.5

# One-shot rental shape, from the compile A/B and quality A/B phase logs.
DPH = 0.481
OCCUPANCY_H = 0.40          # create -> destroyed, including the render
STORAGE_USD = 0.017
# Occupancy EXCLUDING both pulls, so a slow host is charged for its own
# slowness instead of being handed the fast host's timeline. Comparing hosts on
# a fixed 0.40h is exactly the error that makes cheap bandwidth look free.
OCCUPANCY_EX_PULL_H = 0.23
SPEED_DERATE = 0.35         # advertised inet_down is a weak predictor; derate hard

TIERS = [0.0, 0.004, 1.0, 2.667, 4.0, 10.0]


def dl_cost(rate_per_tb, gib=BILLED_GIB):
    return gib / GIB * rate_per_tb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", action="store_true",
                    help="also query live offers and bucket them by download tier")
    ap.add_argument("--gpu", default="RTX 5090")
    # Relaxable, because "no host passes $1/TB" is only meaningful if you know
    # whether the ceiling or the other filters did the excluding.
    ap.add_argument("--min-speed", type=float, default=2500)
    ap.add_argument("--max-dph", type=float, default=0.70)
    args = ap.parse_args()

    print("payload (from povision_fp8.sh)")
    for name, gb in MODELS:
        print(f"  {gb:6.2f} GB  {name}")
    print(f"  {'-' * 6}")
    print(f"  {MODELS_GIB:6.2f} GB  models (TOTAL_BYTES_TO_DOWNLOAD)")
    print(f"  {CUDA_DEV_GIB:6.2f} GB  CUDA dev libs (still pulled, background since 8547f71)")
    print(f"  {BILLED_GIB:6.2f} GB  ACTUALLY BILLED (instance 46593521 invoice)")
    print(f"  {IMAGE_GIB:6.2f} GB  container image - billed as time only, $0")
    print(f"  {UPLOAD_GIB:6.2f} GB  video home (upload direction)\n")

    rental = DPH * OCCUPANCY_H + STORAGE_USD
    print(f"one-shot rental floor: ${DPH}/hr x {OCCUPANCY_H}h + ${STORAGE_USD} "
          f"storage = ${rental:.3f}\n")

    hdr = f"{'$/TB down':>10} {'download $':>11} {'total $':>9} {'dl share':>9} {'vs $0/TB':>9}"
    print(hdr)
    print("-" * len(hdr))
    base = rental
    for t in TIERS:
        d = dl_cost(t)
        total = rental + d
        print(f"{t:>10.3f} {d:>11.4f} {total:>9.3f} {d / total * 100:>8.1f}% "
              f"{total - base:>+9.3f}")

    # Where the download charge equals the rental - the point past which
    # bandwidth, not the GPU, is what you are buying.
    breakeven = rental / (BILLED_GIB / GIB)
    print(f"\ndownload charge == rental charge at ${breakeven:.2f}/TB")
    print(f"upload is {BILLED_GIB / UPLOAD_GIB:.0f}x smaller: at $4.00/TB it costs "
          f"${dl_cost(4.0, UPLOAD_GIB):.4f} - never worth a filter")
    print(f"deferring the CUDA libs entirely would cut {CUDA_DEV_GIB:.2f} GB = "
          f"${dl_cost(2.667, CUDA_DEV_GIB):.3f} at $2.667/TB (time, not money)")

    if not args.market:
        return 0

    print("\n" + "=" * 72)
    print("live market, bucketed by download price")
    q = (f"gpu_name={args.gpu.replace(' ', '_')} num_gpus=1 rentable=true "
         f"gpu_ram >= 24 disk_space >= 60 "
         f"inet_down >= {args.min_speed} dph_total <= {args.max_dph}")
    print(f"query: inet_down >= {args.min_speed} Mb/s, dph_total <= ${args.max_dph}")
    out = subprocess.run(["vastai", "search", "offers", q, "--raw", "-d"],
                         capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        print(out.stderr.strip(), file=sys.stderr)
        return 1
    offers = json.loads(out.stdout)
    print(f"{len(offers)} on-demand offers matching speed/ram/disk/dph\n")

    priced = []
    for o in offers:
        tb = (o.get("inet_down_cost") or 0.0) * GIB
        dph = o.get("dph_total") or 0.0
        spd = (o.get("inet_down") or 0.0) * SPEED_DERATE
        # Both pulls burn rental: the image first, then the models.
        pull_gb = (BILLED_GIB + IMAGE_GIB) * GIB**3 / 1000**3  # GiB -> gigabit basis
        dl_min = (pull_gb * 8 * 1000 / spd / 60) if spd else 999.0
        rent = dph * (OCCUPANCY_EX_PULL_H + dl_min / 60.0) + STORAGE_USD
        priced.append({"tb": tb, "dph": dph, "spd": o.get("inet_down") or 0.0,
                       "dl_min": dl_min, "rent": rent,
                       "total": rent + dl_cost(tb),
                       "loc": (o.get("geolocation") or "?")[:22]})

    buckets = {"free (<$0.01/TB)": [], "$0.01-1/TB": [], "$1-3/TB": [],
               "$3-5/TB": [], ">$5/TB": []}
    for p in priced:
        tb = p["tb"]
        key = ("free (<$0.01/TB)" if tb < 0.01 else
               "$0.01-1/TB" if tb <= 1 else
               "$1-3/TB" if tb <= 3 else
               "$3-5/TB" if tb <= 5 else ">$5/TB")
        buckets[key].append(p)

    hdr = (f"{'bucket':<18} {'n':>4} {'best $':>8} {'rental':>7} {'bandw':>7} "
           f"{'$/hr':>6} {'$/TB':>7} {'Mb/s':>7} {'dl_min':>7}  loc")
    print(hdr)
    print("-" * len(hdr))
    for k, v in buckets.items():
        if not v:
            print(f"{k:<18} {0:>4}       -")
            continue
        b = min(v, key=lambda p: p["total"])
        print(f"{k:<18} {len(v):>4} {b['total']:>8.3f} {b['rent']:>7.3f} "
              f"{dl_cost(b['tb']):>7.3f} {b['dph']:>6.3f} {b['tb']:>7.3f} "
              f"{b['spd']:>7.0f} {b['dl_min']:>7.1f}  {b['loc']}")

    # What a ceiling actually buys. This is the question: does capping $/TB
    # lower the bill, or just move the cost from bandwidth into rental time?
    print(f"\n{'ceiling':>10} {'n pass':>7} {'best total $':>13} {'best host':>44}")
    print("-" * 78)
    for cap in (0.01, 1.0, 2.0, 3.0, 5.0, 999.0):
        ok = [p for p in priced if p["tb"] <= cap]
        if not ok:
            print(f"{cap:>10.2f} {0:>7}   nothing passes")
            continue
        b = min(ok, key=lambda p: p["total"])
        print(f"{cap:>10.2f} {len(ok):>7} {b['total']:>13.3f}   "
              f"${b['dph']:.3f}/hr ${b['tb']:.3f}/TB {b['spd']:.0f}Mb/s "
              f"{b['loc']}")

    # The decisive question is not "is free bandwidth cheaper" - obviously it
    # is - but "how slow may a free-bandwidth host be before the extra rental
    # time it burns eats the saving". Answer that directly, because it is what
    # decides whether MIN_INET_DOWN_SPEED is protecting us or costing us.
    cheap = [p for p in priced if p["tb"] < 0.01]
    paid = [p for p in priced if p["tb"] >= 0.01]
    if cheap and paid:
        c = min(cheap, key=lambda p: p["total"])
        f = min(paid, key=lambda p: p["total"])
        # rent = dph*(occ + dl/60) + storage; the free host pays $0 bandwidth.
        budget_min = ((f["total"] - STORAGE_USD) / c["dph"]
                      - OCCUPANCY_EX_PULL_H) * 60
        pull_gb = (BILLED_GIB + IMAGE_GIB) * GIB**3 / 1000**3
        need = pull_gb * 8 * 1000 / (budget_min * 60) if budget_min > 0 else 0
        print("\nbreak-even, free bandwidth vs the cheapest paid host:")
        print(f"  free  ${c['dph']:.3f}/hr {c['spd']:>5.0f}Mb/s adv "
              f"$0.000/TB -> ${c['total']:.3f}  ({c['dl_min']:.1f} min pull)")
        print(f"  paid  ${f['dph']:.3f}/hr {f['spd']:>5.0f}Mb/s adv "
              f"${f['tb']:.3f}/TB -> ${f['total']:.3f}  ({f['dl_min']:.1f} min pull)")
        print(f"  saving if it performs as modelled: "
              f"${f['total'] - c['total']:.3f}/video "
              f"({(f['total'] - c['total']) / f['total'] * 100:.0f}%)")
        print(f"  the free host may spend up to {budget_min:.0f} min downloading "
              f"and still cost no more -")
        print(f"  that is {need:.0f} Mb/s achieved, only "
              f"{need / c['spd'] * 100:.0f}% of its advertised {c['spd']:.0f}.")
        print(f"  but it makes you wait {c['dl_min'] - f['dl_min']:+.1f} min; at "
              f"$0.02/min that wait is worth "
              f"${(c['dl_min'] - f['dl_min']) * 0.02:.3f}, which is the whole trade.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
