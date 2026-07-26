#!/usr/bin/env python3
"""
Build a manifest mapping every generated video to the config that produced it.

ComfyUI names outputs by a global counter (InfiniteTalk_00001, _00002, ...), so
a directory of mp4s carries no information about which experiment each one came
from. That mapping only exists in the run harness's result JSONs, which record
workflow name, run index, wall-clock and the emitted filename together.

Reconstructing it from memory after the instance is destroyed is impossible, so
this reads the result_*.json files and the workflow JSONs and writes the join
out next to the videos.

    build_output_manifest.py --results <dir of result_*.json> \
                             --workflows <dir of *_API.json> \
                             --videos <dir of mp4s> \
                             --out output/MANIFEST.md
"""
import argparse
import glob
import json
import os

# Node ids in the API-format workflows, for pulling the settings that actually
# distinguish these runs from each other.
N_LOADER = "122"
N_SAMPLER = "128"
N_I2V = "192"
N_COMPILE = "177"
N_SWAP = "134"
N_RESIZE = "281"
N_AUDIO = "125"


def load_results(results_dir):
    """Flatten every result record, tagging it with the file it came from."""
    records = []
    for path in sorted(glob.glob(os.path.join(results_dir, "result_*.json"))):
        try:
            data = json.load(open(path))
        except Exception as e:  # noqa: BLE001 - a truncated file should not kill the report
            print(f"  skipping {os.path.basename(path)}: {e}")
            continue
        for rec in data if isinstance(data, list) else [data]:
            records.append(rec)
    return records


def describe_workflow(wf_dir, name):
    """Best-effort settings summary for a workflow name; blank if not found."""
    for cand in (f"{name}_API.json", f"{name}.json"):
        path = os.path.join(wf_dir, cand)
        if os.path.exists(path):
            break
    else:
        return {}

    try:
        wf = json.load(open(path))
    except Exception:  # noqa: BLE001
        return {}

    def inp(nid, key, default=""):
        try:
            return wf[nid]["inputs"][key]
        except (KeyError, TypeError):
            return default

    w, h = inp(N_RESIZE, "width"), inp(N_RESIZE, "height")
    return {
        "res": f"{w}x{h}" if w and h else "",
        "attn": inp(N_LOADER, "attention_mode"),
        "quant": inp(N_LOADER, "quantization"),
        "steps": inp(N_SAMPLER, "steps"),
        "sched": inp(N_SAMPLER, "scheduler"),
        "compile": inp(N_COMPILE, "mode"),
        "swap": inp(N_SWAP, "blocks_to_swap"),
        "window": inp(N_I2V, "frame_window_size"),
        "tiled_vae": inp(N_I2V, "tiled_vae"),
        "audio": inp(N_AUDIO, "audio"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True)
    ap.add_argument("--workflows", required=True)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    records = load_results(args.results)

    # filename -> record. A cached ComfyUI hit can emit the same filename twice;
    # keep the first (real) run rather than silently overwriting it.
    by_file = {}
    for rec in records:
        for out in rec.get("outputs") or []:
            by_file.setdefault(out, rec)

    have = sorted(os.path.basename(p) for p in glob.glob(os.path.join(args.videos, "*.mp4")))

    lines = [
        "# Generated video manifest",
        "",
        "Mapping from ComfyUI's sequential output names to the experiment that",
        "produced each one, joined from the run harness's `result_*.json` files.",
        "",
        "`seconds` is end-to-end wall-clock for that queue item. Run 1 of any pair",
        "pays the torch.compile warmup and is discarded; run 2 is the measured",
        "number. ComfyUI writes both a silent `.mp4` and an `-audio.mp4` per",
        "generation; only the `-audio` one is listed in the harness output.",
        "",
        "| File | Workflow | Run | Seconds | Res | Attn | Steps | Scheduler | Compile | Window | Tiled VAE | Swap | Audio |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    described = {}
    unmatched = []
    for fname in have:
        rec = by_file.get(fname)
        if rec is None:
            # The silent twin of a matched -audio file is expected, not a gap.
            twin = fname.replace(".mp4", "-audio.mp4")
            if twin in by_file:
                rec = by_file[twin]
            else:
                unmatched.append(fname)
                continue

        wfname = rec.get("workflow", "?")
        if wfname not in described:
            described[wfname] = describe_workflow(args.workflows, wfname)
        d = described[wfname]

        lines.append(
            f"| `{fname}` | {wfname} | {rec.get('run','?')} | "
            f"{round(rec.get('seconds',0),1)} | {d.get('res','')} | {d.get('attn','')} | "
            f"{d.get('steps','')} | {d.get('sched','')} | {d.get('compile','')} | "
            f"{d.get('window','')} | {d.get('tiled_vae','')} | {d.get('swap','')} | "
            f"{d.get('audio','')} |")

    if unmatched:
        lines += ["", "## Not matched to any result record", "",
                  "These predate the run harness (queued by hand during the OOM "
                  "investigation), so no config was recorded for them:", ""]
        lines += [f"- `{f}`" for f in unmatched]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    open(args.out, "w").write("\n".join(lines) + "\n")
    print(f"{len(have)} videos, {len(have)-len(unmatched)} matched -> {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
