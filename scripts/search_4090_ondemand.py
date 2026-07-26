#!/usr/bin/env python3
"""
Non-interactive 4090 on-demand price check.

Reuses interactive_search_vastai's query criteria and cost model rather than
restating them, so the numbers here mean the same thing as what that script
shows: base $/hr plus container storage plus the one-time model download,
against the same PCIe / bandwidth / disk floors.

The interactive script is curses-based and needs a TTY, and its GPU allowlist
covers the whole 4090-or-better family. This narrows to RTX 4090 only and prints
a table.

    python3 scripts/search_4090_ondemand.py
    python3 scripts/search_4090_ondemand.py --any-gpu    # whole allowlist
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import interactive_search_vastai as iv  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--any-gpu", action="store_true",
                    help="do not narrow to RTX 4090")
    ap.add_argument("--limit", type=int, default=15)
    args = ap.parse_args()

    offers = iv.run_vastai_search("on-demand")
    if not args.any_gpu:
        offers = [o for o in offers if "RTX 4090" in (o.get("gpu_name") or "")]

    if not offers:
        print("no matching on-demand offers")
        print(f"criteria: gpu_ram>={iv.MIN_GPU_RAM}GB disk>={iv.MIN_DISK_SPACE}GB "
              f"inet_down>={iv.MIN_INET_DOWN_SPEED}Mb/s pcie_bw>={iv.MIN_PCIE_BW}GB/s "
              f"dph_total<={iv.MAX_DPH}")
        return 0

    for o in offers:
        o["_total"] = iv.calculate_total_cost(o)
    offers.sort(key=lambda o: o["_total"])

    print(f"criteria: gpu_ram>={iv.MIN_GPU_RAM}GB disk>={iv.MIN_DISK_SPACE}GB "
          f"inet_down>={iv.MIN_INET_DOWN_SPEED}Mb/s pcie_bw>={iv.MIN_PCIE_BW}GB/s "
          f"dph_total<=${iv.MAX_DPH}")
    print(f"cost model: rental + {iv.CONTAINER_SIZE_GB}GB storage + "
          f"{iv.DATA_DOWNLOAD_GB}GB one-time download\n")

    hdr = (f"{'id':>10} {'gpu':<10} {'n':>2} {'$/hr':>7} {'1h tot':>7} "
           f"{'down Mb/s':>10} {'pcie':>6} {'disk':>6} {'$/GB dn':>8} {'loc':<14} {'rel':>5}")
    print(hdr)
    print("-" * len(hdr))
    for o in offers[:args.limit]:
        print(f"{o.get('id', 0):>10} "
              f"{(o.get('gpu_name') or '')[:10]:<10} "
              f"{o.get('num_gpus', 0):>2} "
              f"{o.get('dph_total', 0):>7.3f} "
              f"{o['_total']:>7.3f} "
              f"{o.get('inet_down', 0):>10.0f} "
              f"{o.get('pcie_bw', 0):>6.1f} "
              f"{o.get('disk_space', 0):>6.0f} "
              f"{o.get('inet_down_cost', 0):>8.4f} "
              f"{(o.get('geolocation') or '?')[:14]:<14} "
              f"{o.get('reliability2', 0):>5.3f}")

    cheap = offers[0]
    print(f"\ncheapest: {cheap.get('id')} at ${cheap.get('dph_total'):.3f}/hr "
          f"(${cheap['_total']:.3f} for hour 1 incl. storage + download)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
