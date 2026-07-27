#!/usr/bin/env python3
"""
Is the output actually as smooth as its container claims?

    scripts/check_smoothness.py path/to/video.mp4

VHS_VideoCombine stamps whatever frame_rate the workflow says (25) onto whatever
the model produced. If the generator's effective rate is lower, or if windows
repeat frames at their seams, the container still reports 25 fps and the file
still plays - it just does not look smooth. This measures three things the
container cannot tell you:

- unique frames, via mpdecimate: duplicates mean the effective rate is lower
  than the nominal one
- per-frame change, to locate stalls (a run of near-identical frames)
- video length against the source audio, since a mismatch means the muxed rate
  is not the rate the frames were generated for
"""
import json
import os
import re
import subprocess
import sys


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def probe(path, args):
    out = run(["ffprobe", "-v", "error"] + args + [path])
    return out.stdout.strip()


def unique_frames(path):
    """Frames surviving mpdecimate; equal to total when every frame differs."""
    out = run(["ffmpeg", "-v", "error", "-i", path,
               "-vf", "mpdecimate", "-fps_mode", "vfr",
               "-f", "rawvideo", "-y", os.devnull,
               "-progress", "-", "-nostats"])
    m = re.findall(r"frame=\s*(\d+)", out.stdout)
    return int(m[-1]) if m else None


def frame_deltas(path):
    """Mean per-frame luma difference, and how many frames barely changed."""
    out = run(["ffmpeg", "-v", "error", "-i", path,
               "-vf", "tblend=all_mode=difference,signalstats,"
                      "metadata=print:key=lavfi.signalstats.YAVG:file=-",
               "-f", "null", "-"])
    vals = [float(v) for v in
            re.findall(r"lavfi\.signalstats\.YAVG=([0-9.]+)", out.stdout)]
    return vals


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    path = sys.argv[1]
    audio = sys.argv[2] if len(sys.argv) > 2 else None

    total = int(probe(path, ["-select_streams", "v", "-count_frames",
                             "-show_entries", "stream=nb_read_frames",
                             "-of", "csv=p=0"]) or 0)
    dur = float(probe(path, ["-show_entries", "format=duration",
                             "-of", "csv=p=0"]) or 0)
    rate = probe(path, ["-select_streams", "v", "-show_entries",
                        "stream=r_frame_rate", "-of", "csv=p=0"])

    uniq = unique_frames(path)
    vals = frame_deltas(path)

    report = {
        "file": os.path.basename(path),
        "container_fps": rate,
        "duration_s": round(dur, 3),
        "frames_total": total,
        "frames_unique": uniq,
        "duplicate_frames": (total - uniq) if uniq is not None else None,
        "effective_fps": round(uniq / dur, 2) if uniq and dur else None,
    }

    if vals:
        vals_sorted = sorted(vals)
        report["frame_delta_mean"] = round(sum(vals) / len(vals), 4)
        report["frame_delta_min"] = round(vals_sorted[0], 4)
        report["frame_delta_p05"] = round(vals_sorted[len(vals) // 20], 4)
        report["frame_delta_max"] = round(vals_sorted[-1], 4)
        # A frame that barely changed from its predecessor reads as a stall even
        # when mpdecimate does not call it an exact duplicate.
        thresh = report["frame_delta_mean"] * 0.25
        report["near_static_frames"] = sum(1 for v in vals if v < thresh)

    if audio:
        adur = float(probe(audio, ["-show_entries", "format=duration",
                                   "-of", "csv=p=0"]) or 0)
        report["audio_s"] = round(adur, 3)
        report["video_minus_audio_s"] = round(dur - adur, 3)
        if adur:
            report["fps_implied_by_audio"] = round(total / adur, 3)

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
