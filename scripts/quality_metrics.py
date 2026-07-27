#!/usr/bin/env python3
"""
Objective quality metrics over the whole clip, not four hand-picked frames.

    scripts/quality_metrics.py

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

CLIPS = {
    "6step_dpm_sdpa": "output/vast_45930851/InfiniteTalk_00012-audio.mp4",
    "4step_distill_sdpa": "output/vast_45930851/InfiniteTalk_00006-audio.mp4",
    "4step_distill_sage": "output/vast_45930851/InfiniteTalk_00014-audio.mp4",
}


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


def main():
    rows = {}
    for name, path in CLIPS.items():
        rows[name] = {
            "edge_energy": edge_energy(path),
            "temporal_delta": temporal_delta(path),
        }
        print(f"{name:22} edge={rows[name]['edge_energy']} "
              f"flicker={rows[name]['temporal_delta']}", file=sys.stderr)

    pair = ssim(CLIPS["4step_distill_sdpa"], CLIPS["4step_distill_sage"])
    print(f"SSIM 4step sdpa vs sage: {pair}", file=sys.stderr)

    # Deliberately also computed, and deliberately labelled as divergence: the
    # number is low and that is expected, not a quality verdict.
    div = ssim(CLIPS["6step_dpm_sdpa"], CLIPS["4step_distill_sage"])
    print(f"SSIM 6step vs 4step (trajectory divergence, NOT quality): {div}",
          file=sys.stderr)

    print(json.dumps({"clips": rows,
                      "ssim_sdpa_vs_sage": pair,
                      "ssim_6step_vs_4step_divergence": div}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
