#!/usr/bin/env python3
"""
Turn a provisioning.log into a critical-path report.

Everything in povision_fp8.sh runs concurrently, so the total time at the bottom
of the log tells you nothing about what to optimize. This reconstructs the single
timeline from the [PHASE] markers, ranks the per-file [DL_TIME] entries, and
reports the SageAttention wheel-vs-source outcome — i.e. the three things that
decide how long an instance takes to become usable.

Usage:

    python3 scripts/parse_provision_log.py provisioning.log
    python3 scripts/parse_provision_log.py provisioning.log --json
"""
import argparse
import json
import re
import sys

PHASE_RE = re.compile(r"\[PHASE\]\s+\+(\d+)m(\d+)s\s+(.*)")
DL_RE = re.compile(r"\[DL_TIME\]\s+(\S+)\s+([\d.]+)s\s+(\d+)MB\s+(\d+)MB/s")
WHEEL_RE = re.compile(r"\[SAGE_WHEEL\]\s+(.*)")
VERDICT_RE = re.compile(r"probe verdict=(\d+)")
BUILD_RE = re.compile(r"SageAttention build complete\. Duration:\s+(\d+)m\s+(\d+)s")
TOTAL_RE = re.compile(r"Total provisioning time:\s+(\d+)m\s+(\d+)s")


def parse(text):
    report = {"phases": [], "downloads": [], "sage": {}, "totals": {}}

    for line in text.splitlines():
        m = PHASE_RE.search(line)
        if m:
            minutes, seconds, label = int(m.group(1)), int(m.group(2)), m.group(3).strip()
            report["phases"].append({"at_s": minutes * 60 + seconds, "label": label})
            continue

        m = DL_RE.search(line)
        if m:
            report["downloads"].append(
                {
                    "file": m.group(1),
                    "seconds": float(m.group(2)),
                    "mb": int(m.group(3)),
                    "mb_per_s": int(m.group(4)),
                }
            )
            continue

        m = WHEEL_RE.search(line)
        if m:
            report["sage"].setdefault("wheel_log", []).append(m.group(1).strip())
            v = VERDICT_RE.search(line)
            if v:
                report["sage"]["probe_verdict"] = int(v.group(1))
            continue

        m = BUILD_RE.search(line)
        if m:
            report["sage"]["source_build_s"] = int(m.group(1)) * 60 + int(m.group(2))
            continue

        m = TOTAL_RE.search(line)
        if m:
            report["totals"]["provisioning_s"] = int(m.group(1)) * 60 + int(m.group(2))

    # Which SageAttention path actually ran. The wheel is only credited when the
    # probe returned 0 — a wheel that installed but failed the probe is a failure.
    if report["sage"].get("probe_verdict") == 0:
        report["sage"]["path"] = "WHEEL"
    elif "source_build_s" in report["sage"] or "probe_verdict" in report["sage"]:
        report["sage"]["path"] = "SOURCE_BUILD"
    else:
        report["sage"]["path"] = "UNKNOWN"

    return report


def fmt(seconds):
    return f"{int(seconds) // 60}m{int(seconds) % 60:02d}s"


def render(report):
    out = []

    out.append("=" * 68)
    out.append("PROVISIONING TIMELINE")
    out.append("=" * 68)
    phases = report["phases"]
    if not phases:
        out.append("  no [PHASE] markers found — is this log from before the")
        out.append("  instrumentation landed on master?")
    for i, p in enumerate(phases):
        # The gap to the NEXT marker is the cost of this step.
        gap = phases[i + 1]["at_s"] - p["at_s"] if i + 1 < len(phases) else None
        gap_str = f"  (+{fmt(gap)})" if gap is not None else ""
        out.append(f"  {fmt(p['at_s']):>8}  {p['label']}{gap_str}")

    dls = sorted(report["downloads"], key=lambda d: d["seconds"], reverse=True)
    if dls:
        out.append("")
        out.append("=" * 68)
        out.append("DOWNLOADS (slowest first)")
        out.append("=" * 68)
        out.append(f"  {'file':<52}{'time':>8}{'MB/s':>8}")
        for d in dls:
            out.append(f"  {d['file'][:52]:<52}{fmt(d['seconds']):>8}{d['mb_per_s']:>8}")
        total_s = sum(d["seconds"] for d in dls)
        total_mb = sum(d["mb"] for d in dls)
        out.append("  " + "-" * 66)
        avg = int(total_mb / total_s) if total_s else 0
        out.append(f"  {'TOTAL':<52}{fmt(total_s):>8}{avg:>8}")

    out.append("")
    out.append("=" * 68)
    out.append("SAGEATTENTION")
    out.append("=" * 68)
    sage = report["sage"]
    out.append(f"  path            : {sage.get('path', 'UNKNOWN')}")
    if "probe_verdict" in sage:
        verdict = sage["probe_verdict"]
        out.append(f"  probe verdict   : {verdict} ({'usable' if verdict == 0 else 'REJECTED'})")
    if "source_build_s" in sage:
        out.append(f"  source build    : {fmt(sage['source_build_s'])}")
    for line in sage.get("wheel_log", []):
        out.append(f"    | {line}")

    out.append("")
    out.append("=" * 68)
    out.append("VERDICT")
    out.append("=" * 68)

    total = report["totals"].get("provisioning_s")
    if total:
        out.append(f"  total provisioning: {fmt(total)}")

    # Name the actual bottleneck rather than leaving it to be eyeballed.
    dl_total = sum(d["seconds"] for d in report["downloads"])
    build_s = report["sage"].get("source_build_s", 0)
    if dl_total or build_s:
        if build_s > dl_total:
            out.append(
                f"  BOTTLENECK: SageAttention source build ({fmt(build_s)}) "
                f"exceeds downloads ({fmt(dl_total)})."
            )
            if report["sage"].get("path") == "SOURCE_BUILD":
                out.append("  -> A working cached wheel would remove this from the critical path.")
                out.append("     Check the [SAGE_WHEEL] lines above for why the wheel was rejected.")
        else:
            out.append(
                f"  BOTTLENECK: downloads ({fmt(dl_total)}) exceed the "
                f"sage build ({fmt(build_s)})."
            )
            out.append("  -> Wheel caching will NOT help. Attack the payload instead:")
            out.append("     bake models into a custom image, or drop unused ones.")

    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logfile", help="path to provisioning.log ('-' for stdin)")
    parser.add_argument("--json", action="store_true", help="emit parsed JSON instead")
    args = parser.parse_args()

    text = sys.stdin.read() if args.logfile == "-" else open(args.logfile, encoding="utf-8", errors="replace").read()
    report = parse(text)

    print(json.dumps(report, indent=2) if args.json else render(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
