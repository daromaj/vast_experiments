#!/usr/bin/env python3
"""
Turn an e2e_oneshot.sh run directory into a phase table.

    scripts/e2e_report.py output/e2e_20260727T063000Z

Durations come from phases.tsv, which e2e_oneshot.sh writes as the run happens.
Deriving them from log timestamps afterwards would silently attribute waiting on
a poll loop to whatever phase happened to log last.
"""
import glob
import json
import os
import sys

# Label pairs, in order, with the human name for the interval between them.
SPANS = [
    ("create_issued", "ssh_up", "create -> ssh reachable"),
    ("ssh_up", "comfy_up", "provisioning (models, nodes, sage)"),
    ("comfy_up", "uploaded", "upload workflow + assets"),
    ("uploaded", "rendered", "render (cold cache)"),
    ("rendered", "downloaded", "download outputs"),
    ("downloaded", "destroyed", "destroy"),
]


def fmt(seconds):
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}m {s:02d}s"


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    stamps = {}
    order = []
    with open(os.path.join(run_dir, "phases.tsv")) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            epoch, label = line.split("\t", 1)
            stamps[label] = int(epoch)
            order.append(label)

    out = ["# One-shot rental, end to end", ""]

    inst = os.path.join(run_dir, "instance_id")
    if os.path.exists(inst):
        out.append(f"instance: {open(inst).read().strip()}")

    out += ["", "| Phase | Duration |", "|---|---|"]
    total = 0.0
    for a, b, name in SPANS:
        if a in stamps and b in stamps:
            d = stamps[b] - stamps[a]
            total += d
            out.append(f"| {name} | {fmt(d)} |")
    if "create_issued" in stamps and order:
        wall = stamps[order[-1]] - stamps["create_issued"]
        out.append(f"| **total (billed wall clock)** | **{fmt(wall)}** |")

    # The render number the harness itself reports, as a cross-check on the
    # phase stamp - the stamp also contains ssh and process startup.
    for path in glob.glob(os.path.join(run_dir, "results", "*.json")):
        try:
            data = json.load(open(path))
        except Exception:
            continue
        for r in data if isinstance(data, list) else []:
            out.append("")
            out.append(f"harness: {r.get('workflow')} status={r.get('status')} "
                       f"seconds={r.get('seconds')} outputs={r.get('outputs')}")

    peak = os.path.join(run_dir, "peak_vram_mib.txt")
    if os.path.exists(peak):
        vals = [int(x) for x in open(peak).read().split() if x.isdigit()]
        if vals:
            out.append(f"peak VRAM: {max(vals)} MiB")

    vids = sorted(glob.glob(os.path.join(run_dir, "videos", "*.mp4")))
    if vids:
        out.append("")
        out.append("videos:")
        for v in vids:
            out.append(f"  {os.path.basename(v)}  {os.path.getsize(v)/1e6:.1f} MB")

    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
