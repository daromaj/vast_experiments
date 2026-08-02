#!/usr/bin/env python3
"""
Objective quality metrics over the whole clip, not four hand-picked frames.

    scripts/quality_metrics.py                       # the July 2026 sweep clips
    scripts/quality_metrics.py a=one.mp4 b=two.mp4   # any pair, in order

Two different questions need two different metrics:

- **Same sampler, different attention kernel** (4step sdpa vs 4step sage): the
  trajectory is identical, so SSIM is meaningful and should be near 1. This is
  the check that SageAttention is numerically harmless rather than merely fast.
- **Different sampler / step count** (6step dpm++_sde vs 4step distill): the
  denoising trajectories diverge from the first step, so SSIM measures
  divergence, not quality, and a low value proves nothing. Compare
  trajectory-independent statistics instead: edge energy (sharpness proxy) and
  frame-to-frame change (temporal stability / flicker proxy).
"""
import json
import re
import subprocess
import sys

DEFAULT_CLIPS = {
    "6step_dpm_sdpa": "output/vast_45930851/InfiniteTalk_00012-audio.mp4",
    "4step_distill_sdpa": "output/vast_45930851/InfiniteTalk_00006-audio.mp4",
    "4step_distill_sage": "output/vast_45930851/InfiniteTalk_00014-audio.mp4",
}
# The default set answers two questions at once, so its SSIM pairs are chosen by
# name. Given clips on the command line there is only one pair to form, and the
# honest label for it depends on whether the sampler changed - which the caller
# knows and this script does not. So it is reported as "pairwise" with the
# warning attached, rather than silently claiming to measure quality.
SSIM_PAIRS_DEFAULT = [
    ("4step_distill_sdpa", "4step_distill_sage", "ssim_sdpa_vs_sage",
     "same sampler, different attention kernel - meaningful, expect ~1"),
    ("6step_dpm_sdpa", "4step_distill_sage", "ssim_6step_vs_4step_divergence",
     "different sampler AND step count - trajectory divergence, NOT quality"),
]


def ffmpeg(args):
    return subprocess.run(["ffmpeg", "-hide_banner", "-nostats"] + args,
                          capture_output=True, text=True)


def mean_metadata(path, vf, key):
    """Average one signalstats value across every frame of a clip."""
    out = ffmpeg(["-i", path, "-vf",
                  f"{vf},metadata=print:key={key}:file=-",
                  "-f", "null", "-"])
    vals = [float(m) for m in re.findall(rf"{re.escape(key)}=([0-9.]+)", out.stdout)]
    return (sum(vals) / len(vals)) if vals else None


def edge_energy(path):
    """Mean luma after a Sobel pass: higher = more edge detail = sharper."""
    return mean_metadata(path, "sobel,signalstats",
                         "lavfi.signalstats.YAVG")


def temporal_delta(path):
    """Mean luma of consecutive-frame absolute difference: higher = more flicker."""
    return mean_metadata(path, "tblend=all_mode=difference,signalstats",
                         "lavfi.signalstats.YAVG")


def ssim(a, b):
    out = ffmpeg(["-i", a, "-i", b, "-lavfi", "ssim", "-f", "null", "-"])
    m = re.search(r"All:([0-9.]+)", out.stderr)
    return float(m.group(1)) if m else None


def parse_args(argv):
    if not argv:
        return DEFAULT_CLIPS, SSIM_PAIRS_DEFAULT
    clips = {}
    for arg in argv:
        if "=" not in arg:
            sys.exit(f"expected label=path, got {arg!r}")
        label, path = arg.split("=", 1)
        clips[label] = path
    if len(clips) != 2:
        sys.exit("give exactly two label=path clips, or none for the defaults")
    a, b = list(clips)
    return clips, [(a, b, f"ssim_{a}_vs_{b}",
                    "pairwise - only meaningful if the sampler and step count "
                    "are identical; otherwise it measures trajectory divergence")]


def main():
    clips, ssim_pairs = parse_args(sys.argv[1:])

    rows = {}
    for name, path in clips.items():
        rows[name] = {
            "edge_energy": edge_energy(path),
            "temporal_delta": temporal_delta(path),
        }
        print(f"{name:26} edge={rows[name]['edge_energy']} "
              f"flicker={rows[name]['temporal_delta']}", file=sys.stderr)

    out = {"clips": rows}
    for a, b, key, note in ssim_pairs:
        val = ssim(clips[a], clips[b])
        out[key] = val
        print(f"SSIM {a} vs {b}: {val}  [{note}]", file=sys.stderr)

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
