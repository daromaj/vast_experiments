#!/usr/bin/env python3
"""
Find 5090s with cheap egress, which the main search cannot surface.

interactive_search_vastai.py filters on inet_down >= 5000 Mb/s, and on a one-shot
rental that is backwards: the 34 GB payload is paid once per video, so $/GB
matters, and every host clearing 5000 charged $2.60-$10.00/TB.

The opposite mistake is just as expensive. Ranking on money alone picked a
1311 Mb/s host at $1.30/TB that sat in `loading` for 14 minutes, never reached
ssh and produced nothing - it was still pulling the 9.5 GB container image, which
vast bills as time but not as egress, so a $-ranked search is blind to it.

So this ranks on both: modelled one-shot cost plus the wait priced at
--time-value $/min, under hard ceilings on egress, speed and total cost.

    scripts/search_cheap_egress.py                 # 5090, on-demand
    scripts/search_cheap_egress.py --gpu "RTX 4090" --max-cost-per-tb 1.5
    scripts/search_cheap_egress.py --time-value 0  # rank on money only
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import interactive_search_vastai as iv  # noqa: E402

MODELS_GB = 34.0
# The container image is pulled from Docker Hub on every fresh rental, before
# provisioning starts and before ssh answers. vastai/comfy:v0.28.0-cuda-12.9-py312
# is 9.53 GB compressed. It costs time but NOT money: the e2e run billed
# bwd = 34.4 GB, which is the models alone, so vast does not charge egress on the
# image pull. Ranking on $ therefore cannot see it at all - which is exactly how
# a host that spent 14 minutes stuck in `loading` got ranked third-best.
IMAGE_GB = 9.5
# Advertised inet_down is what the host claims; the pull rarely approaches it.
# Machine 69187 advertised 1311 Mb/s, which puts the 9.5 GB image at under a
# minute - it had not finished after 14 minutes, so under 10% of line rate.
# Machine 54134 at 1699 Mb/s had ssh up in 52 s. Advertised speed is a weak
# predictor, so derate it hard and keep a floor under --min-speed rather than
# trusting the number.
# Applies to hosts below the median-share figure. Rests on a SINGLE measured
# point - a 1699 Mb/s host delivered 1638 Mb/s, 96.4% - and is hedged down to
# 0.90 because n=1. It is the weakest constant in this file.
SPEED_DERATE = 0.90
# Measured, six runs. Regenerate with scripts/calibrate_bandwidth.py, which
# reads provisioning.log directly instead of trusting this comment:
#
#     advertised   window   achieved     % of advertised
#     1699 Mb/s     166 s   1638 Mb/s    96.4%
#     7318 Mb/s      86 s   3161 Mb/s    43.2%
#     7398 Mb/s     205 s   1408 Mb/s    19.0%
#     7944 Mb/s      98 s   2774 Mb/s    34.9%
#     8021 Mb/s     159 s   1710 Mb/s    21.3%
#     9135 Mb/s     261 s   1106 Mb/s    12.1%
#
# CORRECTION 2026-08-02: this block previously listed 7398->1299, 1699->1371 and
# 7944->799 and the constants were fitted to them. Those numbers were wrong.
# They used the ABSOLUTE offset of "[PHASE] downloads finished" as the download
# duration, but downloads start only after apt finishes. Under the old
# provisioning script the blocking 2 GB CUDA install ran first, so the 7944 Mb/s
# run was charged 3m32s of apt against its model pull - 340 s rather than the
# real 98 s - and reported 799 Mb/s for a host that actually delivered 2774.
#
# WHAT THE CORRECTED DATA SAYS: among hosts advertising >=3000 Mb/s, achieved
# throughput spans 1106-3161 Mb/s and does NOT track the advertised figure. The
# 9135 Mb/s host was the slowest of the six; the 7318 Mb/s host was the fastest.
# Above the floor, advertised inet_down carries essentially no information.
#
# WHY: inet_down is the MACHINE's uplink, shared by every instance on it, so a
# rental gets roughly link/tenants. Big advertised numbers belong to multi-GPU
# rigs with several renters on one fat pipe; a modest single-tenant host hands
# over nearly all of what it claims (the 1699 Mb/s box: 96.4%).
#
# So this is an observed MEDIAN SHARE, not a physical ceiling, and with a 2.9x
# spread it is a poor predictor for any individual host. It is kept because
# ranking needs a number: without a cap the score pays --time-value for minutes
# that never get saved and reliably outbids a cheap host for an expensive fast
# one. That is how the 2026-08-02 run rented a $0.481/hr 7398 Mb/s box at
# $2.667/TB when a $0.334/hr 1678 Mb/s box at $0.000/TB was available and
# provisioned in the same time. The floor plus a price ranking is what actually
# does the work here; this constant only stops the model overpaying for speed.
OBSERVED_MEDIAN_SHARE_MBPS = 1700
# Rental time for one 58 s video end to end EXCLUDING every download: boot, the
# rest of provisioning, a cold-cache render, output upload, teardown. Download
# time is added per offer from its link speed. Folding it into a single constant
# would rank a 1300 Mb/s host as if it pulled 43 GB as fast as a 8000 Mb/s one,
# which is exactly the error that makes cheap egress look free.
OCCUPANCY_EX_DOWNLOAD_H = 0.23


def search(gpu, min_speed, max_dph):
    query = (f"gpu_name={gpu.replace(' ', '_')} num_gpus=1 rentable=true "
             f"gpu_ram >= 24 disk_space >= 60 "
             f"inet_down >= {min_speed} dph_total <= {max_dph}")
    cmd = ["vastai", "search", "offers", query, "--raw", "-d"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        print("vastai search timed out", file=sys.stderr)
        return []
    if out.returncode != 0:
        print(out.stderr.strip(), file=sys.stderr)
        return []
    try:
        offers = json.loads(out.stdout)
    except json.JSONDecodeError:
        print("could not parse vastai output", file=sys.stderr)
        return []

    # Tag the market explicitly. The query passes -d, so these are on-demand
    # offers, but the raw JSON does not always carry instance_type and
    # iv.create_instance defaults a missing value to 'bid' - which silently
    # rents an interruptible box that can be reclaimed mid-render.
    for o in offers:
        o["instance_type"] = "on-demand"
    return offers


def main():
    ap = argparse.ArgumentParser()
    # agent_vastai.py cannot rent these hosts: its create path re-runs the main
    # search, which filters inet_down >= 5000, and a cheap-egress host is
    # typically well under that. So creating has to live here, next to the
    # search that can actually see them. The instance itself is still built by
    # iv.create_instance, so image, env and disk stay single-sourced.
    ap.add_argument("--create", type=int, metavar="MACHINE_ID",
                    help="rent this machine_id from the ranked results")
    # Naming a machine_id read from an earlier listing loses a race often enough
    # to be the normal case, not the exception: the cheap-egress hosts are the
    # contended ones, and two attempts in a row failed within a second of the
    # host appearing in a listing. This ranks and rents in one process, so there
    # is no window between choosing and asking.
    ap.add_argument("--create-best", action="store_true",
                    help="rent the top-ranked offer from this same search")
    ap.add_argument("--skip", type=int, nargs="*", default=[],
                    help="machine_ids to pass over (already tried, or known bad)")
    ap.add_argument("--gpu", default="RTX 5090")
    # A ceiling, not a target. $1.50 was the target and it routinely returned an
    # empty list once --min-speed rose to 2500, which makes --create-best fail
    # rather than rent something sane. 3.0 admits the common $2.60/TB tier and
    # lets the score prefer cheap egress instead of the filter mandating it.
    ap.add_argument("--max-cost-per-tb", type=float, default=3.0,
                    help="egress ceiling in $/TB (vast reports $/GB)")
    # 1000 -> 2500 -> 1500. A 1311 Mb/s host failed to pull the image in 14
    # minutes, so a floor has to exist. But 2500 was set before the ceiling
    # above was measured, and it excluded every free-bandwidth host on the
    # market for a benefit of under two cents: on 2026-08-02 it left just THREE
    # candidates at --max-cost-per-tb 3.0, against eight at 1500. Three is thin
    # enough that --create-best failing is a live outcome, and this search has
    # already lost the offer race twice.
    #
    # 1500 keeps us above the 1311 Mb/s host that actually stalled, while
    # admitting the 1678-1730 Mb/s hosts that carry $0.000/TB bandwidth.
    ap.add_argument("--min-speed", type=float, default=1500,
                    help="Mb/s floor; stay above ~1300 or the pull dominates")
    ap.add_argument("--max-cost", type=float, default=0.40,
                    help="hard ceiling on modelled one-shot $, so a fast host "
                         "still cannot charge whatever it likes")
    # $0.02/min says: five cents is worth about two and a half minutes. Raise it
    # when the video is wanted now, drop it toward 0 to rank on money alone.
    ap.add_argument("--time-value", type=float, default=0.02,
                    help="$ per minute of waiting, used to rank cost vs speed")
    ap.add_argument("--max-dph", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=15)
    args = ap.parse_args()

    offers = search(args.gpu, args.min_speed, args.max_dph)
    ceiling = args.max_cost_per_tb / 1000.0

    rows = []
    for o in offers:
        cost_gb = o.get("inet_down_cost") or 0.0
        if cost_gb > ceiling:
            continue
        dph = o.get("dph_total") or 0.0
        # Egress is billed on the models only - the image pull is free in $.
        download = cost_gb * MODELS_GB
        # Minutes spent pulling the image and then the models, at a derated
        # fraction of the advertised link speed.
        speed = min((o.get("inet_down") or 0.0) * SPEED_DERATE,
                    OBSERVED_MEDIAN_SHARE_MBPS)
        dl_min = ((IMAGE_GB + MODELS_GB) * 8 * 1000 / speed / 60) if speed else 999.0
        rental = dph * (OCCUPANCY_EX_DOWNLOAD_H + dl_min / 60.0)
        total = rental + download
        if args.max_cost and total > args.max_cost:
            continue
        # Rank on money AND time together by pricing the wait. Ranking on $
        # alone picked a host that sat in `loading` for 14 minutes and produced
        # nothing; ranking on time alone invites being gouged for a marginally
        # faster link. --time-value makes the trade explicit and tunable, and
        # the hard ceilings (--min-speed, --max-cost-per-tb, --max-cost) stop it
        # from buying speed at any price.
        eta_min = OCCUPANCY_EX_DOWNLOAD_H * 60 + dl_min
        score = total + eta_min * args.time_value
        rows.append((score, eta_min, total, download, rental, dl_min, o))

    rows.sort(key=lambda r: r[0])

    if args.create is not None or args.create_best:
        if args.create_best:
            cands = [r for r in rows if r[6].get("machine_id") not in args.skip]
        else:
            cands = [r for r in rows if r[6].get("machine_id") == args.create]
        if not cands:
            what = "no offer left" if args.create_best else f"machine {args.create}"
            print(f"{what} in the current ranked results (taken, or outside the "
                  "egress/speed ceiling) - re-run search", file=sys.stderr)
            return 1
        # Walk the ranking: an offer can vanish between ranking and asking, and
        # the next one down is still a good rental.
        for score, eta_min, total, dl, rent, dl_min, offer in cands:
            print(f"machine {offer['machine_id']} offer {offer['id']}: "
                  f"{offer.get('gpu_name')} ${offer.get('dph_total'):.3f}/hr "
                  f"egress ${(offer.get('inet_down_cost') or 0) * 1000:.2f}/TB "
                  f"{offer.get('inet_down', 0):.0f}Mb/s {offer.get('geolocation')} "
                  f"rel={offer.get('reliability2', 0):.3f} -> "
                  f"eta {eta_min:.0f}min, one-shot ${total:.3f}")
            if iv.create_instance(offer):
                return 0
            if not args.create_best:
                return 1
            print("  that offer did not take - trying the next", file=sys.stderr)
        return 1

    hdr = (f"{'machine':>8} {'$/hr':>6} {'$/TB':>6} {'down':>6} {'dl_min':>7} "
           f"{'eta_min':>7} {'rental':>7} {'egress':>7} {'1-shot':>7} "
           f"{'score':>7} {'loc':<16} {'rel':>5}")
    print(f"criteria: {args.gpu}, egress <= ${args.max_cost_per_tb}/TB, "
          f"inet_down >= {args.min_speed}Mb/s, one-shot <= ${args.max_cost}")
    print(f"model: {OCCUPANCY_EX_DOWNLOAD_H}h rental + per-offer pull of "
          f"{IMAGE_GB}GB image + {MODELS_GB}GB models at {SPEED_DERATE:.0%} of "
          f"advertised speed")
    print(f"score = one-shot $ + eta_min x ${args.time_value}/min "
          f"(lower is better)\n")
    print(hdr)
    print("-" * len(hdr))
    if not rows:
        print("nothing passed - raise --max-cost-per-tb, --max-cost, "
              "or lower --min-speed")
        return 0
    for score, eta_min, total, dl, rent, dl_min, o in rows[:args.limit]:
        print(f"{o.get('machine_id', 0):>8} "
              f"{o.get('dph_total', 0):>6.3f} "
              f"{(o.get('inet_down_cost') or 0) * 1000:>6.2f} "
              f"{o.get('inet_down', 0):>6.0f} "
              f"{dl_min:>7.1f} {eta_min:>7.1f} "
              f"{rent:>7.3f} {dl:>7.3f} {total:>7.3f} {score:>7.3f} "
              f"{(o.get('geolocation') or '?')[:16]:<16} "
              f"{o.get('reliability2', 0):>5.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
