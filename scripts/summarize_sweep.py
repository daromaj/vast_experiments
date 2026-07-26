#!/usr/bin/env python3
"""
Render a markdown results table from run_july_tests.py result files.

Transcribing timings by hand into README.md is how a 130.6s warmup ends up in the
docs when the JSON says 646.5s. Everything quotable comes from the files.

Peak VRAM is read from the .vram sidecars run_matrix.sh writes, and is reported
as *reserved*, not required: PyTorch's caching allocator keeps what it has taken,
so a variant pinned at the device limit reports the device limit whatever its
true working set. It separates "had headroom" from "had none", nothing finer.

    summarize_sweep.py --results notes/run_results/4090/json \
                       --vram notes/run_results/4090/vram
"""
import argparse
import glob
import json
import os
import re


def load(results_dir):
    """{variant: {run: record}}"""
    out = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "result_*.json"))):
        try:
            data = json.load(open(path))
        except Exception as e:
            print(f"<!-- skipped {path}: {e} -->")
            continue
        if not isinstance(data, list):
            continue
        for r in data:
            out.setdefault(r.get("workflow", "?"), {})[r.get("run")] = r
    return out


def vram_peaks(vram_dir):
    peaks = {}
    if not vram_dir or not os.path.isdir(vram_dir):
        return peaks
    for path in glob.glob(os.path.join(vram_dir, "*.vram")):
        name = os.path.basename(path)[: -len(".vram")]
        try:
            peaks[name] = int(open(path).read().strip())
        except Exception:
            pass
    return peaks


def swap_of(name):
    """q4_swap12_untiled -> 12, for ordering the curve rather than the filenames."""
    m = re.search(r"swap(\d+)", name)
    return int(m.group(1)) if m else -1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True)
    ap.add_argument("--vram")
    ap.add_argument("--device-limit-mib", type=int, default=24080,
                    help="reserved reading that means 'pinned at the ceiling'")
    args = ap.parse_args()

    runs = load(args.results)
    peaks = vram_peaks(args.vram)

    print("| variant | swap | cold (run 1) | warm (run 2) | peak reserved | result |")
    print("|---|---|---|---|---|---|")
    for name in sorted(runs, key=lambda n: (swap_of(n), n)):
        r1, r2 = runs[name].get(1), runs[name].get(2)
        swap = swap_of(name)
        cold = f"{r1['seconds']:.1f} s" if r1 and r1.get("seconds") else "—"
        ok = r2 and r2.get("status") == "success"
        warm = f"**{r2['seconds']:.1f} s**" if ok else "—"
        pk = peaks.get(name)
        pk_s = f"{pk:,} MiB" if pk else "—"
        if pk and pk >= args.device_limit_mib - 16:
            pk_s += " (pinned)"
        if ok:
            verdict = "success"
        elif r1 and r1.get("node_error", {}) and \
                "OutOfMemory" in str(r1["node_error"].get("exception_type")):
            verdict = "**OOM**"
        else:
            verdict = (r1 or {}).get("status", "?")
        print(f"| {name} | {swap if swap >= 0 else '—'} | {cold} | {warm} | {pk_s} | {verdict} |")


if __name__ == "__main__":
    raise SystemExit(main())
