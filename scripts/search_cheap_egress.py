#!/usr/bin/env python3
"""
Find 5090s with cheap egress, which the main search cannot surface.

interactive_search_vastai.py sorts by a one-hour total that already includes the
34 GB model download, but it first filters on inet_down >= 5000 Mb/s. On a
one-shot rental that filter is backwards: the download is paid once per video, so
$/GB dominates, and a host at 2000 Mb/s and $0.001/GB beats one at 8000 Mb/s and
$0.0026/GB on both cost and (because the download is only ~34 GB) barely loses on
time.

    scripts/search_cheap_egress.py                 # 5090, on-demand
    scripts/search_cheap_egress.py --gpu "RTX 4090" --max-cost-per-tb 1.5

Prints the offers sorted by total one-shot cost: rental for the estimated
occupancy plus the download.
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import interactive_search_vastai as iv  # noqa: E402

DOWNLOAD_GB = 34.0
# Rental time for one 58 s video end to end EXCLUDING the model download:
# boot, the rest of provisioning, a cold-cache render, output download, teardown.
# The download is added per offer from its link speed. Folding it into a single
# constant would rank a 1300 Mb/s host as if it pulled 34 GB as fast as a
# 8000 Mb/s one, which is exactly the error that makes cheap egress look free.
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
    ap.add_argument("--max-cost-per-tb", type=float, default=1.5,
                    help="egress ceiling in $/TB (vast reports $/GB)")
    ap.add_argument("--min-speed", type=float, default=1000,
                    help="Mb/s floor, well below the main search's 5000")
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
        download = cost_gb * DOWNLOAD_GB
        # Minutes spent pulling 34 GB at the advertised link speed. Advertised,
        # so treat it as a floor: real provisioning has never hit line rate.
        speed = o.get("inet_down") or 0.0
        dl_min = (DOWNLOAD_GB * 8 * 1000 / speed / 60) if speed else 0.0
        rental = dph * (OCCUPANCY_EX_DOWNLOAD_H + dl_min / 60.0)
        rows.append((rental + download, download, rental, dl_min, o))

    rows.sort(key=lambda r: r[0])

    if args.create is not None or args.create_best:
        if args.create_best:
            cands = [r for r in rows if r[4].get("machine_id") not in args.skip]
        else:
            cands = [r for r in rows if r[4].get("machine_id") == args.create]
        if not cands:
            what = "no offer left" if args.create_best else f"machine {args.create}"
            print(f"{what} in the current ranked results (taken, or outside the "
                  "egress/speed ceiling) - re-run search", file=sys.stderr)
            return 1
        # Walk the ranking: an offer can vanish between ranking and asking, and
        # the next one down is still a good rental.
        for total, dl, rent, dl_min, offer in cands:
            print(f"machine {offer['machine_id']} offer {offer['id']}: "
                  f"{offer.get('gpu_name')} ${offer.get('dph_total'):.3f}/hr "
                  f"egress ${(offer.get('inet_down_cost') or 0) * 1000:.2f}/TB "
                  f"{offer.get('inet_down', 0):.0f}Mb/s {offer.get('geolocation')} "
                  f"rel={offer.get('reliability2', 0):.3f} -> one-shot ${total:.3f}")
            if iv.create_instance(offer):
                return 0
            if not args.create_best:
                return 1
            print("  that offer did not take - trying the next", file=sys.stderr)
        return 1

    hdr = (f"{'machine':>8} {'$/hr':>6} {'$/TB':>6} {'down':>6} {'dl_min':>7} "
           f"{'rental':>7} {'egress':>7} {'1-shot':>7} {'loc':<16} {'rel':>5}")
    print(f"criteria: {args.gpu}, egress <= ${args.max_cost_per_tb}/TB, "
          f"inet_down >= {args.min_speed}Mb/s")
    print(f"one-shot model: {OCCUPANCY_EX_DOWNLOAD_H}h rental + per-offer "
          f"download time for {DOWNLOAD_GB}GB + egress\n")
    print(hdr)
    print("-" * len(hdr))
    if not rows:
        print("nothing under that egress ceiling - raise --max-cost-per-tb")
        return 0
    for total, dl, rent, dl_min, o in rows[:args.limit]:
        print(f"{o.get('machine_id', 0):>8} "
              f"{o.get('dph_total', 0):>6.3f} "
              f"{(o.get('inet_down_cost') or 0) * 1000:>6.2f} "
              f"{o.get('inet_down', 0):>6.0f} "
              f"{dl_min:>7.1f} "
              f"{rent:>7.3f} {dl:>7.3f} {total:>7.3f} "
              f"{(o.get('geolocation') or '?')[:16]:<16} "
              f"{o.get('reliability2', 0):>5.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
