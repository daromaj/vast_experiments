#!/usr/bin/env python3
"""
Pin the host-ranking model against the runs that calibrated it.

    scripts/test_host_ranking.py

No network, no rental: synthetic offers built from hosts we actually rented, so
the two defects fixed on 2026-08-02 cannot come back quietly.

  1. Download time must SATURATE. Advertised link speed above ~1600 Mb/s does
     not shorten the pull - measured, a 1699 Mb/s host beat a 7944 Mb/s one -
     and a model that scales with the advertised number pays --time-value for
     minutes that are never saved.
  2. Given that, a cheap slow-ish host must OUTRANK an expensive fast one. This
     is the concrete failure: on 2026-08-02 the search rented a $0.481/hr
     7398 Mb/s box at $2.667/TB while a $0.334/hr 1678 Mb/s box at $0.000/TB
     sat unrented, provisioning in the same time for half the money.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import interactive_search_vastai as iv  # noqa: E402
import search_cheap_egress as ce  # noqa: E402


def offer(mid, dph, mbps, cost_per_tb, **kw):
    o = {"id": mid, "machine_id": mid, "dph_total": dph, "inet_down": mbps,
         "inet_down_cost": cost_per_tb / 1024.0, "gpu_name": "RTX 5090",
         "gpu_ram": 32768, "num_gpus": 1, "instance_type": "on-demand",
         "reliability2": 0.99, "geolocation": "Test, XX"}
    o.update(kw)
    return o


# The three hosts that matter, from real listings.
POLAND = offer(144163, 0.481, 7398, 2.667)   # what the search actually rented
HONGKONG = offer(140190, 0.334, 1678, 0.000)  # what it should have rented
STALLER = offer(69187, 0.300, 1311, 0.000)   # never pulled 9.5GB in 14 minutes

FAILURES = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        if detail:
            print(f"      {detail}")
        FAILURES.append(name)


def dl_min_ce(o):
    """search_cheap_egress's download-time term, mirrored."""
    speed = min((o.get("inet_down") or 0.0) * ce.SPEED_DERATE,
                ce.OBSERVED_MEDIAN_SHARE_MBPS)
    return (ce.IMAGE_GB + ce.MODELS_GB) * 8 * 1000 / speed / 60


def main():
    # --- 1. saturation ------------------------------------------------------
    fast, mid = dl_min_ce(POLAND), dl_min_ce(HONGKONG)
    check("download time saturates: 7398 Mb/s is not much faster than 1678",
          abs(fast - mid) < 1.0,
          f"7398 -> {fast:.1f} min, 1678 -> {mid:.1f} min "
          f"(gap {abs(fast - mid):.1f} min, want < 1.0)")

    slow = dl_min_ce(offer(1, 0.3, 700, 0.0))
    check("...but a genuinely slow host is still modelled as slower",
          slow > mid + 2.0,
          f"700 Mb/s -> {slow:.1f} min vs 1678 -> {mid:.1f} min")

    # iv uses the same ceiling via calculate_total_cost, which stamps
    # download_minutes onto the offer as a side effect.
    a, b = dict(POLAND), dict(HONGKONG)
    iv.calculate_total_cost(a)
    iv.calculate_total_cost(b)
    check("interactive picker saturates too",
          abs(a["download_minutes"] - b["download_minutes"]) < 1.0,
          f"{a['download_minutes']:.1f} vs {b['download_minutes']:.1f} min")

    # --- 2. the ranking that cost real money --------------------------------
    def ce_score(o):
        cost_gb = o["inet_down_cost"]
        dl = cost_gb * ce.MODELS_GB
        dm = dl_min_ce(o)
        rental = o["dph_total"] * (ce.OCCUPANCY_EX_DOWNLOAD_H + dm / 60.0)
        eta = ce.OCCUPANCY_EX_DOWNLOAD_H * 60 + dm
        return rental + dl + eta * 0.02

    pl, hk = ce_score(POLAND), ce_score(HONGKONG)
    check("cheap free-bandwidth host outranks the expensive fast one",
          hk < pl,
          f"Hong Kong ${hk:.3f} vs Poland ${pl:.3f} (lower must win)")

    iv_pl = iv.calculate_total_cost(dict(POLAND))
    iv_hk = iv.calculate_total_cost(dict(HONGKONG))
    check("...and in the interactive picker's cost model as well",
          iv_hk < iv_pl, f"HK ${iv_hk:.3f} vs PL ${iv_pl:.3f}")

    # --- 3. the floor still excludes the host that actually stalled ---------
    check("speed floor still rejects the 1311 Mb/s host that stalled 14 min",
          STALLER["inet_down"] < iv.MIN_INET_DOWN_SPEED,
          f"floor is {iv.MIN_INET_DOWN_SPEED}, staller advertised "
          f"{STALLER['inet_down']}")
    check("...while admitting the free-bandwidth hosts",
          HONGKONG["inet_down"] >= iv.MIN_INET_DOWN_SPEED,
          f"floor {iv.MIN_INET_DOWN_SPEED} vs {HONGKONG['inet_down']} Mb/s")

    # --- 4. both scripts agree on the floor ---------------------------------
    import argparse
    ap = argparse.ArgumentParser()
    ce_default = None
    for action in ce.main.__wrapped__.__code__.co_consts if False else []:
        pass
    # Read the declared default straight out of the parser ce.main builds.
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "search_cheap_egress.py")).read()
    import re
    m = re.search(r'"--min-speed", type=float, default=(\d+)', src)
    ce_default = int(m.group(1)) if m else None
    del ap
    check("both searches use the same speed floor",
          ce_default == iv.MIN_INET_DOWN_SPEED,
          f"search_cheap_egress default={ce_default}, "
          f"interactive MIN_INET_DOWN_SPEED={iv.MIN_INET_DOWN_SPEED}")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all host-ranking checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
