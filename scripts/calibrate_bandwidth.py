#!/usr/bin/env python3
"""
Achieved download throughput vs advertised inet_down, one row per real run.

    scripts/calibrate_bandwidth.py

Every rental produces a free data point: create.log records the host's
advertised inet_down, and provisioning.log's [PHASE] lines bracket the model
pull. This turns those into the numbers that provision_table.py's model is
fitted to, so the fit is auditable instead of asserted.

The payload is the model set plus the CUDA dev libs, because the sage build
downloads those DURING the same window and they compete for the same link.
Counting only the models understates achieved throughput by roughly 6%.
"""
import argparse
import glob
import os
import re
import sys

MODELS_BYTES = 33978384650      # measured, see egress_cost.py
CUDA_BYTES = 2093 * 1000**2     # apt reports decimal MB


def parse_run(run_dir):
    create = os.path.join(run_dir, "create.log")
    prov = os.path.join(run_dir, "provisioning.log")
    if not (os.path.exists(create) and os.path.exists(prov)):
        return None

    m = re.search(r"([\d.]+)Mb/s", open(create).read())
    if not m:
        return None
    advertised = float(m.group(1))

    text = open(prov, errors="replace").read()
    start = re.search(r"\[PHASE\] \+(\d+)m(\d+)s .*downloads starting", text)
    end = re.search(r"\[PHASE\] \+(\d+)m(\d+)s downloads finished", text)
    if not (start and end):
        return None
    t0 = int(start.group(1)) * 60 + int(start.group(2))
    t1 = int(end.group(1)) * 60 + int(end.group(2))
    secs = t1 - t0
    if secs <= 0:
        return None

    # The CUDA fetch only counts if it landed inside the download window.
    payload = MODELS_BYTES
    cuda = re.search(r"\[SAGE_BUILD\] Fetched (\d+) MB", text)
    if cuda:
        payload += int(cuda.group(1)) * 1000**2

    achieved = payload * 8 / secs / 1e6      # Mb/s, decimal
    return dict(run=os.path.basename(run_dir), advertised=advertised,
                secs=secs, achieved=achieved,
                pct=100.0 * achieved / advertised if advertised else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", help="run dirs (default: output/e2e_*)")
    args = ap.parse_args()

    dirs = args.runs or sorted(glob.glob(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "output", "e2e_*")))
    rows = [r for r in (parse_run(d) for d in dirs) if r]
    if not rows:
        print("no runs with both create.log and a complete provisioning.log",
              file=sys.stderr)
        return 1

    print(f"{'advertised':>11} {'window':>8} {'achieved':>9} {'% of adv':>9}  run")
    print("-" * 62)
    for r in sorted(rows, key=lambda r: r["advertised"]):
        print(f"{r['advertised']:>11.0f} {r['secs']:>7}s {r['achieved']:>9.0f} "
              f"{r['pct']:>8.1f}%  {r['run']}")

    fast = [r for r in rows if r["advertised"] >= 3000]
    if fast:
        vals = sorted(r["achieved"] for r in fast)
        mid = vals[len(vals) // 2] if len(vals) % 2 else \
            (vals[len(vals) // 2 - 1] + vals[len(vals) // 2]) / 2
        print(f"\nhosts advertising >=3000 Mb/s (n={len(fast)}): "
              f"delivered {min(vals):.0f}-{max(vals):.0f} Mb/s, median {mid:.0f}")
        print("  -> this median is what OBSERVED_MEDIAN_SHARE should be set to; the "
              "spread is why it is a median share, not a cap")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
