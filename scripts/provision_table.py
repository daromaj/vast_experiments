#!/usr/bin/env python3
"""
Time-to-provision and one-shot cost, across advertised speed x download price.

    scripts/provision_table.py                # the grid
    scripts/provision_table.py --offers       # the same model applied to live offers

The point of the grid is to make one thing obvious before renting anything:
above roughly 1500 Mb/s advertised, buying more link speed buys almost no time,
because the host's NIC stopped being the bottleneck. Download PRICE keeps
mattering all the way up. So the two axes are not symmetric, and the intuition
that a faster host is a better host is worth about $0.10 a video.

CALIBRATION - three measured runs, from output/*/provisioning.log, comparing
the [PHASE] "downloads finished" offset against the host's advertised inet_down
recorded in create.log:

    advertised   downloads finished   achieved     % of advertised
    7398 Mb/s    3m29s (209 s)        1299 Mb/s    17.6%
    1699 Mb/s    3m18s (198 s)        1371 Mb/s    80.7%
    7944 Mb/s    5m40s (340 s)         799 Mb/s    10.1%

The 1699 Mb/s host finished FASTER than the 7944 Mb/s one. Achieved throughput
sits in a band around 800-1400 Mb/s whatever the link claims, so the model is a
derate with a hard ceiling rather than a plain percentage:

    achieved = min(advertised x LINK_DERATE, PIPELINE_CEILING)

n=3. Treat the ceiling as the shape of the thing, not a precise constant - the
7944 Mb/s run came in well under it and nothing in the log explains why. The
conclusion that survives regardless: past ~1700 Mb/s advertised, extra speed is
not what you are buying.
"""
import argparse
import json
import subprocess
import sys

GIB = 1024

# Billed payload, from the invoice (see egress_cost.py for the derivation).
BILLED_GIB = 34.6
IMAGE_GIB = 9.5          # container image: costs time, never money

LINK_DERATE = 0.80       # fits the 1699 Mb/s point at 80.7%
PIPELINE_CEILING = 1300  # Mb/s, the band the two fast hosts actually delivered

# create -> ssh_up was 38 s on the 2026-08-02 run because the host already had
# the image. A cache MISS is the single largest unpredictable cost in the whole
# timeline - one host never finished pulling 9.5 GB in 14 minutes - and it is
# not predicted by advertised speed either. Modelled separately and flagged,
# never folded into a single "provisioning time" number.
IMAGE_CACHED_MIN = 0.6
POST_DOWNLOAD_MIN = 1.5  # nodes settle, sage wheel probe, WANOPT patch, restart
RENDER_MIN = 11.0        # 58 s clip, cold inductor cache (arm A: 659 s)
TEARDOWN_MIN = 1.0

STORAGE_USD = 0.017

SPEEDS = [1000, 1500, 2000, 3000, 5000, 8000]
PRICES = [0.0, 1.0, 2.667, 4.0, 6.667, 10.0]


def achieved_mbps(advertised):
    return min(advertised * LINK_DERATE, PIPELINE_CEILING)


def download_min(advertised, gib):
    gbit = gib * GIB**3 / 1000**3 * 8  # GiB -> decimal gigabit
    mbps = achieved_mbps(advertised)
    return gbit * 1000 / mbps / 60 if mbps else 999.0


def provision_min(advertised, image_cached=True):
    """create -> ready to render."""
    img = IMAGE_CACHED_MIN if image_cached else download_min(advertised, IMAGE_GIB)
    return img + download_min(advertised, BILLED_GIB) + POST_DOWNLOAD_MIN


def one_shot(advertised, price_per_tb, dph, image_cached=True):
    hours = (provision_min(advertised, image_cached)
             + RENDER_MIN + TEARDOWN_MIN) / 60.0
    return dph * hours + STORAGE_USD + BILLED_GIB / GIB * price_per_tb


def grid(dph, image_cached):
    print(f"\nassumptions: ${dph:.3f}/hr, {BILLED_GIB} GB billed payload, "
          f"{RENDER_MIN:.0f} min render, image "
          f"{'CACHED on host' if image_cached else 'PULLED (9.5 GB)'}")
    print(f"achieved = min(advertised x {LINK_DERATE}, {PIPELINE_CEILING} Mb/s)\n")

    w = 14
    print(f"{'advertised':>11} {'provision':>10} {'achieved':>9} |"
          + "".join(f"{('$%.3f/TB' % p):>{w}}" for p in PRICES))
    print(f"{'Mb/s':>11} {'min':>10} {'Mb/s':>9} |"
          + "".join(f"{'total $':>{w}}" for _ in PRICES))
    print("-" * (34 + w * len(PRICES)))
    for s in SPEEDS:
        cells = "".join(f"{one_shot(s, p, dph, image_cached):>{w}.3f}"
                        for p in PRICES)
        print(f"{s:>11} {provision_min(s, image_cached):>10.1f} "
              f"{achieved_mbps(s):>9.0f} |{cells}")

    # The two axes, isolated. This is the number that decides the policy.
    slow, fast = SPEEDS[0], SPEEDS[-1]
    t = (provision_min(slow, image_cached) - provision_min(fast, image_cached))
    print(f"\n  {slow} -> {fast} Mb/s advertised buys {t:.1f} min "
          f"(= ${t / 60 * dph:.3f} of rental at ${dph:.3f}/hr)")
    d = (PRICES[-1] - PRICES[0]) * BILLED_GIB / GIB
    print(f"  $0 -> ${PRICES[-1]:.0f}/TB costs ${d:.3f} — "
          f"{d / max(t / 60 * dph, 1e-9):.1f}x the entire speed benefit")


def offers(gpu, dph_cap, image_cached):
    q = (f"gpu_name={gpu.replace(' ', '_')} num_gpus=1 rentable=true "
         f"gpu_ram >= 24 disk_space >= 60 inet_down >= 500 "
         f"dph_total <= {dph_cap}")
    out = subprocess.run(["vastai", "search", "offers", q, "--raw", "-d"],
                         capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        print(out.stderr.strip(), file=sys.stderr)
        return 1
    rows = []
    for o in json.loads(out.stdout):
        adv = o.get("inet_down") or 0.0
        tb = (o.get("inet_down_cost") or 0.0) * GIB
        dph = o.get("dph_total") or 0.0
        rows.append((one_shot(adv, tb, dph, image_cached),
                     provision_min(adv, image_cached), adv, tb, dph,
                     o.get("machine_id"), (o.get("geolocation") or "?")[:18],
                     o.get("reliability2") or 0.0))
    rows.sort()
    hdr = (f"{'total $':>8} {'prov min':>9} {'Mb/s':>7} {'$/TB':>7} {'$/hr':>6} "
           f"{'bandw $':>8} {'rel':>6} {'machine':>8}  loc")
    print("\n" + hdr)
    print("-" * len(hdr))
    for t, pm, adv, tb, dph, mid, loc, rel in rows[:20]:
        print(f"{t:>8.3f} {pm:>9.1f} {adv:>7.0f} {tb:>7.3f} {dph:>6.3f} "
              f"{BILLED_GIB / GIB * tb:>8.3f} {rel:>6.3f} {mid:>8}  {loc}")
    print(f"\n{len(rows)} offers; cheapest is ${rows[0][0]:.3f}, "
          f"most expensive shown ${rows[min(19, len(rows) - 1)][0]:.3f}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dph", type=float, default=0.45,
                    help="rental $/hr to price the grid at")
    ap.add_argument("--image-pull", action="store_true",
                    help="model a Docker Hub cache MISS (adds the 9.5 GB image)")
    ap.add_argument("--offers", action="store_true",
                    help="apply the model to live offers instead of a grid")
    ap.add_argument("--gpu", default="RTX 5090")
    ap.add_argument("--max-dph", type=float, default=1.20)
    args = ap.parse_args()

    cached = not args.image_pull
    grid(args.dph, cached)
    if args.offers:
        return offers(args.gpu, args.max_dph, cached)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
