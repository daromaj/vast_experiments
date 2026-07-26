#!/usr/bin/env python3
"""
Non-interactive vast.ai search + create, for agent/script use.

interactive_search_vastai.py is the source of truth for search criteria, the cost
model and the create command; it is also curses-based and needs a TTY, so it
cannot be driven from a script or an unattended run. This exposes the same logic
as a plain CLI and imports it rather than restating it, so the two cannot drift.

    # what is available
    agent_vastai.py search
    agent_vastai.py search --gpu "RTX 4090"
    agent_vastai.py search --gpu "RTX 5090" --max-dph 0.55 --limit 5
    agent_vastai.py search --type both --json

    # rent one (machine_id, from the table's `machine` column)
    agent_vastai.py create 19205
    agent_vastai.py create 19205 --dry-run

Pass --machine-id to `create`, not an offer id. Offer ids are per-slot and rotate
as slots get allocated, so an id read from one search is routinely stale by the
time the next command runs - which looks exactly like "the host was taken" when
the machine is in fact still free. machine_id identifies the physical box.

Cost model (from interactive_search_vastai): base $/hr + container storage for
CONTAINER_SIZE_GB + a one-time DATA_DOWNLOAD_GB model download. The download term
is what reorders the list - a $0.35/hr host with $0.029/GB egress costs more in
hour one than a $0.38/hr host with $0.0026/GB.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import interactive_search_vastai as iv  # noqa: E402


def collect(instance_type, gpu, max_dph):
    """Search one or both markets and apply the extra agent-side filters."""
    types = ["on-demand", "bid"] if instance_type == "both" else [instance_type]

    offers = []
    for t in types:
        offers.extend(iv.run_vastai_search(t))

    if gpu:
        offers = [o for o in offers if gpu.lower() in (o.get("gpu_name") or "").lower()]
    if max_dph is not None:
        offers = [o for o in offers if (o.get("dph_total") or 0) <= max_dph]

    for o in offers:
        o["_total"] = iv.calculate_total_cost(o)
    offers.sort(key=lambda o: o["_total"])
    return offers


def criteria_line(max_dph):
    return (f"gpu_ram>={iv.MIN_GPU_RAM}GB disk>={iv.MIN_DISK_SPACE}GB "
            f"inet_down>={iv.MIN_INET_DOWN_SPEED}Mb/s pcie_bw>={iv.MIN_PCIE_BW}GB/s "
            f"dph_total<=${max_dph if max_dph is not None else iv.MAX_DPH}")


def cmd_search(args):
    offers = collect(args.type, args.gpu, args.max_dph)

    if args.json:
        keep = ("id", "machine_id", "host_id", "gpu_name", "num_gpus", "dph_total",
                "_total", "inet_down", "inet_down_cost", "pcie_bw", "disk_space",
                "geolocation", "reliability2", "instance_type", "cuda_max_good")
        print(json.dumps([{k: o.get(k) for k in keep} for o in offers[:args.limit]],
                         indent=2))
        return 0

    print(f"criteria: {criteria_line(args.max_dph)}")
    print(f"cost model: rental + {iv.CONTAINER_SIZE_GB}GB storage + "
          f"{iv.DATA_DOWNLOAD_GB}GB one-time download")
    if not offers:
        print("\nno matching offers - loosen the criteria in interactive_search_vastai.py")
        return 0
    print()

    hdr = (f"{'machine':>8} {'offer':>10} {'gpu':<10} {'n':>2} {'$/hr':>7} {'1h tot':>7} "
           f"{'down':>7} {'pcie':>6} {'disk':>6} {'$/GB':>7} {'type':<10} {'loc':<15} {'rel':>5}")
    print(hdr)
    print("-" * len(hdr))
    for o in offers[:args.limit]:
        print(f"{o.get('machine_id', 0):>8} "
              f"{o.get('id', 0):>10} "
              f"{(o.get('gpu_name') or '')[:10]:<10} "
              f"{o.get('num_gpus', 0):>2} "
              f"{o.get('dph_total', 0):>7.3f} "
              f"{o['_total']:>7.3f} "
              f"{o.get('inet_down', 0):>7.0f} "
              f"{o.get('pcie_bw', 0):>6.1f} "
              f"{o.get('disk_space', 0):>6.0f} "
              f"{o.get('inet_down_cost', 0):>7.4f} "
              f"{o.get('instance_type', ''):<10} "
              f"{(o.get('geolocation') or '?')[:15]:<15} "
              f"{o.get('reliability2', 0):>5.3f}")

    best = offers[0]
    print(f"\nbest by 1h total: machine {best.get('machine_id')} "
          f"({best.get('gpu_name')}, {best.get('geolocation')}) "
          f"${best.get('dph_total'):.3f}/hr, ${best['_total']:.3f} hour one, "
          f"rel={best.get('reliability2', 0):.3f}")
    print(f"rent it with:  python3 scripts/agent_vastai.py create {best.get('machine_id')}")
    return 0


def cmd_create(args):
    offers = collect(args.type, None, None)
    candidates = [o for o in offers if o.get("machine_id") == args.machine_id]
    if not candidates:
        print(f"no {args.type} offer for machine_id={args.machine_id} in the current "
              "result set (taken, or outside the criteria) - re-run search")
        return 1

    # One machine can expose several slots; take the cheapest.
    match = min(candidates, key=lambda o: o.get("dph_total") or 0)
    print(f"machine {match['machine_id']} offer {match['id']}: {match.get('gpu_name')} "
          f"${match.get('dph_total'):.3f}/hr {match.get('inet_down', 0):.0f}Mb/s "
          f"pcie {match.get('pcie_bw', 0):.1f} disk {match.get('disk_space', 0):.0f}GB "
          f"{match.get('geolocation')} rel={match.get('reliability2', 0):.3f}")

    if args.dry_run:
        print("dry run - not creating")
        return 0

    return 0 if iv.create_instance(match) else 1


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search", help="list offers, cheapest total first")
    s.add_argument("--gpu", help='substring filter, e.g. "RTX 4090"')
    s.add_argument("--type", default="on-demand",
                   choices=["on-demand", "bid", "both"])
    s.add_argument("--max-dph", type=float,
                   help=f"override the ${iv.MAX_DPH} cap (search-side cap still applies)")
    s.add_argument("--limit", type=int, default=15)
    s.add_argument("--json", action="store_true", help="machine-readable output")
    s.set_defaults(func=cmd_search)

    c = sub.add_parser("create", help="rent a machine by machine_id")
    c.add_argument("machine_id", type=int)
    c.add_argument("--type", default="on-demand", choices=["on-demand", "bid"])
    c.add_argument("--dry-run", action="store_true")
    c.set_defaults(func=cmd_create)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
