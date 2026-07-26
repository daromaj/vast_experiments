#!/usr/bin/env python3
"""
Derive the full-length render from a tuned short-clip workflow.

Verified against the 5090 run: full58_sage_API.json differed from
s5_sage_untiled_API.json in exactly one input, 125.audio. InfiniteTalk drives
length from the audio and samples fixed 81-frame windows, so nothing else -
window size, block swap, steps, VRAM ceiling - changes with clip duration.
Rebuilding the whole workflow would only risk drift from the config that was
actually measured.

    make_full58.py workflows/generated/sweep4090/q2_swap06_untiled_API.json \
                   --out workflows/generated/full58_4090_API.json
"""
import argparse
import json

N_AUDIO = "125"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="tuned short-clip API workflow")
    ap.add_argument("--audio", default="santa_58s.mp3")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    wf = json.load(open(args.source))
    node = wf.get(N_AUDIO)
    if not node or node.get("class_type") != "LoadAudio":
        raise SystemExit(f"node {N_AUDIO} is not a LoadAudio in {args.source}")

    was = node["inputs"].get("audio")
    node["inputs"]["audio"] = args.audio
    with open(args.out, "w") as fh:
        json.dump(wf, fh, indent=2)
    print(f"{args.source} -> {args.out}\n  {N_AUDIO}.audio {was} -> {args.audio}")


if __name__ == "__main__":
    main()
