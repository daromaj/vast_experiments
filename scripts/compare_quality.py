#!/usr/bin/env python3
"""
Build side-by-side frame strips so the speed winners can be judged on quality.

    scripts/compare_quality.py --out notes/quality

Speed has been measured to death; quality never was. These clips are directly
comparable because every sweep workflow starts from the same base seed and
run_july_tests.py bumps it deterministically (base + run*1000 + 7), so run 2 of
every variant sampled the same noise. Different settings, same seed - which is
the only way a visual diff means anything.

Emits one PNG per timestamp with the variants stacked left to right and labelled,
plus a per-pair absolute-difference map to make small changes visible.
"""
import argparse
import os
import subprocess
import sys

# (label, video path). Run 2 of each variant - the measured run.
VARIANTS = [
    ("6step_dpm_sdpa", "output/vast_45930851/InfiniteTalk_00012-audio.mp4"),
    ("4step_distill_sdpa", "output/vast_45930851/InfiniteTalk_00006-audio.mp4"),
    ("4step_distill_sage", "output/vast_45930851/InfiniteTalk_00014-audio.mp4"),
]

# Spread across the clip: first window, a window boundary, and late drift.
TIMESTAMPS = ["00:00:01", "00:00:03", "00:00:05", "00:00:07"]


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print(" ".join(cmd), file=sys.stderr)
        print(p.stderr[-800:], file=sys.stderr)
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="notes/quality")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    missing = [p for _, p in VARIANTS if not os.path.exists(p)]
    if missing:
        print("missing inputs:\n  " + "\n  ".join(missing), file=sys.stderr)
        return 1

    for ts in TIMESTAMPS:
        tag = ts.replace(":", "")
        frames = []
        for label, path in VARIANTS:
            out = os.path.join(args.out, f"f_{tag}_{label}.png")
            # -ss before -i seeks fast; accurate enough at these offsets and the
            # same seek is applied to every variant, so they stay aligned.
            if not run(["ffmpeg", "-y", "-loglevel", "error", "-ss", ts,
                        "-i", path, "-frames:v", "1", out]):
                return 1
            frames.append((label, out))

        # Label each panel, then stack horizontally.
        labelled = []
        for label, f in frames:
            lab = f.replace(".png", "_lab.png")
            if not run(["ffmpeg", "-y", "-loglevel", "error", "-i", f,
                        "-vf", f"drawtext=text='{label}':x=8:y=8:fontsize=22:"
                               "fontcolor=yellow:box=1:boxcolor=black@0.6",
                        lab]):
                return 1
            labelled.append(lab)

        # JPEG, because these are committed for review and a three-panel PNG
        # sheet is 2 MB against 250 KB for a visually identical JPEG.
        sheet = os.path.join(args.out, f"compare_{tag}.jpg")
        cmd = ["ffmpeg", "-y", "-loglevel", "error"]
        for lab in labelled:
            cmd += ["-i", lab]
        cmd += ["-filter_complex", f"hstack=inputs={len(labelled)}",
                "-q:v", "3", sheet]
        if not run(cmd):
            return 1
        print(f"wrote {sheet}")

        # The per-variant frames only exist to build the sheet.
        for _, f in frames:
            os.remove(f)
        for lab in labelled:
            os.remove(lab)

    # Difference maps: 6-step vs each 4-step, amplified so subtle drift shows.
    base = VARIANTS[0]
    for label, path in VARIANTS[1:]:
        diff = os.path.join(args.out, f"diff_{base[0]}_vs_{label}.png")
        if not run(["ffmpeg", "-y", "-loglevel", "error",
                    "-ss", "00:00:03", "-i", base[1],
                    "-ss", "00:00:03", "-i", path,
                    "-filter_complex",
                    "[0:v][1:v]blend=all_mode=difference,eq=contrast=6",
                    "-frames:v", "1", diff]):
            return 1
        print(f"wrote {diff}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
