#!/usr/bin/env python3
"""
Turn on VHS_VideoCombine.trim_to_audio in the workflows we actually ship.

    scripts/set_trim_to_audio.py            # report only
    scripts/set_trim_to_audio.py --write

Measured on the 58 s render: the output was 1521 frames where the 58.201 s of
audio needs 1455, so the clip ran 2.64 s past the end of the speech with the
character idling. Motion in that tail averages 0.30 against 0.78 for the body of
the clip - it reads as the video hanging after the voice stops.

Only the shipped workflows are touched. The sweep and compile_ab files under
workflows/generated/{sweep,sweep4090,compile_ab} are records of what was
executed, and rewriting them would misrepresent the runs they document.
"""
import argparse
import json
import sys

SHIPPED = [
    "workflows/IT_4090_july2026_4step.json",
    "workflows/IT_4090_july2026_5step.json",
    "workflows/IT_5090_july2026_4step.json",
    "workflows/IT_5090_july2026_5step.json",
    # Used by e2e_oneshot.sh for the rent-render-destroy path.
    "workflows/generated/full/full58_sage_API.json",
]


def patch(obj, changes, path=""):
    """Set trim_to_audio wherever VHS_VideoCombine appears, API or GUI format."""
    if isinstance(obj, dict):
        ct = obj.get("class_type") or obj.get("type")
        if ct == "VHS_VideoCombine":
            inputs = obj.get("inputs")
            # API format: inputs is a dict of values.
            if isinstance(inputs, dict) and "trim_to_audio" in inputs:
                if inputs["trim_to_audio"] is not True:
                    inputs["trim_to_audio"] = True
                    changes.append(f"{path}: inputs.trim_to_audio -> True")
            # GUI format: widgets_values carries the same field by name.
            wv = obj.get("widgets_values")
            if isinstance(wv, dict) and "trim_to_audio" in wv:
                if wv["trim_to_audio"] is not True:
                    wv["trim_to_audio"] = True
                    changes.append(f"{path}: widgets_values.trim_to_audio -> True")
        for k, v in obj.items():
            patch(v, changes, f"{path}/{k}" if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            patch(v, changes, f"{path}[{i}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    total = 0
    for path in SHIPPED:
        try:
            data = json.load(open(path))
        except FileNotFoundError:
            print(f"MISSING {path}", file=sys.stderr)
            continue
        changes = []
        patch(data, changes)
        if not changes:
            print(f"ok      {path} (already true, or no VHS_VideoCombine)")
            continue
        total += len(changes)
        print(f"CHANGE  {path}")
        for c in changes:
            print(f"          {c}")
        if args.write:
            json.dump(data, open(path, "w"), indent=2)

    print(f"\n{total} change(s){'' if args.write else ' - dry run, pass --write'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
