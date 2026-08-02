#!/usr/bin/env python3
"""
Where is the motion? A whole-frame mean cannot tell a head turn from a wobble.

    scripts/motion_regions.py january=A.mp4 current=B.mp4
    scripts/motion_regions.py --heatmap notes/heat january=A.mp4 current=B.mp4

motion_profile.py answers "how much does this clip move". That is not enough to
judge a talking-head render, because the answer aggregates two opposite things:
the subject moving, which is the point, and scenery moving, which is a defect.
A clip can score *higher* on total motion purely because its chair drifts.

Two views, both label-free enough to trust:

- **Heatmap** (--heatmap): per-pixel temporal mean of frame differences,
  amplified. Shows where motion lives without anyone drawing boxes first. This
  is the honest starting point - read it before believing any region below.
- **Regions**: motion / jerk per named box. The boxes are specific to the
  480x832 santa framing used in the January-vs-current A/B; pass --regions to
  redefine them for anything else. Boxes are ffmpeg crop specs, "w:h:x:y".

Caveat that limits what regions can prove: at a subject's silhouette, "the
chair moved" and "the edge of the man moved" occupy the same pixels. A region
straddling that boundary measures both and cannot separate them. Only regions
well clear of the subject - wall, tree, desk - are clean readings of scenery
stability.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from check_smoothness import frame_deltas  # noqa: E402

# 480x832 santa framing, boxes placed against a drawgrid overlay rather than
# guessed. Three tiers, and the order matters when reading the output:
#
#   wall_*/tree/desk  far scenery. The noise floor - whatever these read is
#                     what "static" costs in this codec at this bitrate.
#   mug/chair_post_*  static PROPS clear of the subject. Any motion above the
#                     noise floor here is a defect: chairs do not move.
#   face_mouth/hand_* the subject. Motion here is the point of the render.
#
# An earlier version of this set drew the chair boxes at the silhouette, where
# "the chair moved" and "his shoulder moved" are the same pixels, and it could
# prove nothing. These posts sit clear of him.
DEFAULT_REGIONS = {
    "wall_left_strip": "30:150:0:300",
    "wall_upper": "180:170:0:0",
    "tree_only": "120:260:360:30",
    "desk_carving": "480:110:0:690",
    "mug": "55:110:0:480",
    "chair_post_L": "45:120:30:300",
    "chair_post_R": "36:100:392:310",
    "face_mouth": "120:150:180:230",
    "hand_L": "80:70:60:520",
    "hand_R": "90:70:350:515",
}


def stats(deltas):
    if len(deltas) < 3:
        return None
    d = deltas[1:]  # first delta is against a black frame in some muxes
    jerks = [abs(d[i] - d[i - 1]) for i in range(1, len(d))]
    motion = statistics.fmean(d)
    jerk = statistics.fmean(jerks)
    return {
        "motion": round(motion, 4),
        "jerk": round(jerk, 4),
        "jerk_over_motion": round(jerk / motion, 4) if motion else None,
    }


def heatmap(path, out_png, gain):
    """
    Per-pixel temporal mean of frame differences.

    tmix carries a rolling window, so the LAST output frame is the average over
    the trailing 1024 differences - hence -update with a select on the tail
    rather than -frames:v, which would take the first frame and average almost
    nothing.
    """
    vf = (f"tblend=all_mode=difference,format=gray,tmix=frames=1024,"
          f"lutyuv=y=clip(val*{gain}\\,0\\,255),select='gte(n\\,1450)'")
    p = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                        "-vf", vf, "-update", "1", out_png],
                       capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stderr[-500:], file=sys.stderr)
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heatmap", metavar="DIR",
                    help="also write a per-pixel motion heatmap per clip")
    ap.add_argument("--gain", type=int, default=60,
                    help="heatmap amplification (default 60; try 14 to see "
                         "only the strongest motion)")
    ap.add_argument("--regions",
                    help="name=w:h:x:y,name=w:h:x:y ... overrides the defaults")
    ap.add_argument("clips", nargs="+", metavar="LABEL=PATH")
    args = ap.parse_args()

    if args.regions:
        regions = dict(r.split("=", 1) for r in args.regions.split(","))
    else:
        regions = DEFAULT_REGIONS

    clips = {}
    for c in args.clips:
        if "=" not in c:
            sys.exit(f"expected LABEL=PATH, got {c!r}")
        label, path = c.split("=", 1)
        clips[label] = path

    if args.heatmap:
        os.makedirs(args.heatmap, exist_ok=True)
        for label, path in clips.items():
            out = os.path.join(args.heatmap, f"heat_{label}.png")
            if heatmap(path, out, args.gain):
                print(f"wrote {out}", file=sys.stderr)

    out = {}
    labels = list(clips)
    print(f"{'region':18} " + " ".join(f"{l:>26}" for l in labels),
          file=sys.stderr)
    for name, crop in regions.items():
        out[name] = {}
        cells = []
        for label, path in clips.items():
            s = stats(frame_deltas(path, crop=crop))
            out[name][label] = s
            cells.append("-" if not s else
                         f"m={s['motion']:.3f} j/m={s['jerk_over_motion']:.3f}")
        print(f"{name:18} " + " ".join(f"{c:>26}" for c in cells),
              file=sys.stderr)

    print(json.dumps({"regions": regions, "results": out}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
