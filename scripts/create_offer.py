#!/usr/bin/env python3
"""
Create a vast.ai instance for a known offer id, without the curses UI.

interactive_search_vastai.create_instance() already encodes the correct create
command - image pin, env blob, disk size, ssh/direct flags, and the on-demand vs
bid branch that matters (passing --bid_price to an on-demand offer silently makes
it interruptible). This reuses that function rather than hand-rolling the command,
so an unattended create cannot drift from what the interactive path does.

    python3 scripts/create_offer.py 42288313
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import interactive_search_vastai as iv  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ident", type=int,
                    help="machine_id by default, or offer id with --by-offer-id")
    ap.add_argument("--by-offer-id", action="store_true",
                    help="match the volatile offer id instead of machine_id")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    offers = iv.run_vastai_search("on-demand")

    # Offer ids are per-slot and rotate as slots get allocated, so an id read from
    # one search is routinely stale by the time the next call runs - which looks
    # exactly like "the host was taken". machine_id identifies the physical box and
    # stays put, so that is the default key.
    key = "id" if args.by_offer_id else "machine_id"
    candidates = [o for o in offers if o.get(key) == args.ident]
    if not candidates:
        print(f"no on-demand offer with {key}={args.ident} in the current result set "
              "(taken, or it dropped out of the criteria) - re-run the search")
        return 1

    # One machine can expose several slots; take the cheapest.
    match = min(candidates, key=lambda o: o.get("dph_total") or 0)

    print(f"offer {match['id']}: {match.get('gpu_name')} "
          f"${match.get('dph_total'):.3f}/hr "
          f"{match.get('inet_down', 0):.0f}Mb/s down "
          f"pcie {match.get('pcie_bw', 0):.1f} "
          f"disk {match.get('disk_space', 0):.0f}GB "
          f"{match.get('geolocation')} rel={match.get('reliability2', 0):.3f}")

    if args.dry_run:
        print("dry run - not creating")
        return 0

    return 0 if iv.create_instance(match) else 1


if __name__ == "__main__":
    raise SystemExit(main())
