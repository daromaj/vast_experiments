#!/usr/bin/env python3
"""
Pick the fastest measured sweep variant and render the full-length clip with it.

Runs ON the instance, unattended, straight after the sweep: the long render is
the critical path, so waiting for someone to read a results table before starting
it wastes the most expensive minutes of the rental.

Selection is on run 2 (warm), not run 1 - run 1 is the torch.compile warmup and
says nothing about steady-state speed. Variants that OOMed have no run 2 and drop
out on their own.

    pick_best_and_full58.py --results /workspace/results --sweep /workspace/sweep
"""
import argparse
import glob
import json
import os
import subprocess
import sys

N_AUDIO = "125"


def measured_times(results_dir):
    """{variant: warm seconds} for variants that actually completed run 2."""
    out = {}
    for path in glob.glob(os.path.join(results_dir, "result_q*.json")):
        try:
            data = json.load(open(path))
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        for r in data:
            if r.get("run") == 2 and r.get("status") == "success":
                out[r.get("workflow")] = r.get("seconds")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="/workspace/results")
    ap.add_argument("--sweep", default="/workspace/sweep")
    ap.add_argument("--audio", default="santa_58s.mp3")
    ap.add_argument("--out-workflow", default="/workspace/sweep/full58_4090_API.json")
    ap.add_argument("--out-result", default="/workspace/results/result_full58_4090.json")
    ap.add_argument("--run-timeout", type=float, default=5400)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    times = measured_times(args.results)
    if not times:
        print("FATAL: no variant produced a successful warm run", flush=True)
        return 1

    print("warm (run 2) times:", flush=True)
    for name, secs in sorted(times.items(), key=lambda kv: kv[1]):
        print(f"  {name:<24} {secs:8.1f}s", flush=True)

    best = min(times, key=lambda k: times[k])
    print(f"\nwinner: {best} at {times[best]:.1f}s", flush=True)

    src = os.path.join(args.sweep, f"{best}_API.json")
    wf = json.load(open(src))
    node = wf.get(N_AUDIO)
    if not node or node.get("class_type") != "LoadAudio":
        print(f"FATAL: node {N_AUDIO} is not a LoadAudio in {src}", flush=True)
        return 1
    node["inputs"]["audio"] = args.audio
    with open(args.out_workflow, "w") as fh:
        json.dump(wf, fh, indent=1)
    print(f"wrote {args.out_workflow} from {src} ({N_AUDIO}.audio -> {args.audio})",
          flush=True)

    if args.dry_run:
        return 0

    # One run only: the sweep already warmed the inductor cache for this exact
    # config, so a discarded first run would just pay for the full clip twice.
    here = os.path.dirname(os.path.abspath(__file__))
    cmd = [sys.executable, os.path.join(here, "run_july_tests.py"),
           "--workflows", args.out_workflow,
           "--runs", "1",
           "--run-timeout", str(args.run_timeout),
           "--out", args.out_result]
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
