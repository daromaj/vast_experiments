#!/usr/bin/env python3
"""
Separate "moves less" from "flickers less". The two look identical in a mean.

    scripts/motion_profile.py label=clip.mp4 [label=clip.mp4 ...]

check_smoothness.py reports frame_delta_mean, and july_test.md already records
the trap: a clip whose subject simply moves less scores lower on it, which is
not evidence of better temporal stability. Comparing two samplers on that number
alone cannot tell a smoother render from a deader one.

So report three things per clip:

- motion       mean per-frame luma delta. How much the picture changes at all.
- jerk         mean absolute change *of* that delta between consecutive frames.
               High jerk with the same motion means the change is uneven.
- jerk/motion  roughness normalised for how much is moving, which is the number
               that survives a comparison between clips with different amounts
               of motion. Lower is smoother.

Plus per-decile motion, because a mean hides a clip that starts lively and dies.

mpdecimate is deliberately not used anywhere here: it collapses frames where
only a mouth moves against a still background, which is this workload exactly,
and reports the result as duplicates. Use framemd5 for a real duplicate count.
"""
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from check_smoothness import frame_deltas  # noqa: E402


def profile(path):
    d = frame_deltas(path)
    if len(d) < 3:
        return {"error": f"only {len(d)} frame deltas from {path}"}

    # The first delta is frame 1 against a black frame in some muxes; drop it
    # rather than let a single outlier set the jerk scale.
    d = d[1:]
    jerks = [abs(d[i] - d[i - 1]) for i in range(1, len(d))]
    motion = statistics.fmean(d)
    jerk = statistics.fmean(jerks)

    n = len(d)
    deciles = [round(statistics.fmean(d[i * n // 10:(i + 1) * n // 10]), 3)
               for i in range(10)]

    return {
        "frames_compared": n,
        "motion": round(motion, 4),
        "motion_sd": round(statistics.pstdev(d), 4),
        "jerk": round(jerk, 4),
        "jerk_over_motion": round(jerk / motion, 4) if motion else None,
        "motion_deciles": deciles,
    }


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    out = {}
    for arg in sys.argv[1:]:
        label, path = arg.split("=", 1) if "=" in arg else (arg, arg)
        out[label] = profile(path)
        r = out[label]
        if "error" in r:
            print(f"{label:16} {r['error']}", file=sys.stderr)
            continue
        print(f"{label:16} motion={r['motion']:.4f} jerk={r['jerk']:.4f} "
              f"jerk/motion={r['jerk_over_motion']:.4f}", file=sys.stderr)
        print(f"{'':16} deciles={r['motion_deciles']}", file=sys.stderr)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
