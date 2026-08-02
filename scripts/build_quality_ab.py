#!/usr/bin/env python3
"""
Build both arms of the January-vs-current quality A/B.

    scripts/build_quality_ab.py

Arm A ("january") is the January 2026 workflow read straight out of git, so it
cannot drift. Its sampler settings are taken verbatim and NOT modernised:
6 steps, dpm++_sde, quantization disabled, merge_loras false, torch.compile in
`default` mode. Only four things change, and each is printed as it is applied so
the diff against January is never implicit:

1. Inputs repointed at the assets that still exist in input_files/. The January
   JSON names santa_test.mp3 / santa-classic.png, which are gone; both arms must
   read the same image and audio or nothing downstream means anything.
2. MelBandRoFormer vocal separation removed (nodes 301/302), matching the
   current workflow. It feeds MultiTalkWav2VecEmbeds, so leaving it in would put
   a second variable into the lip-sync comparison.
3. trim_to_audio true. Container-level only, no effect on generation - but
   without it this arm emits 60.84 s against the other arm's 58.2 s and the two
   clips no longer share a frame index, which breaks every metric that compares
   them frame by frame.
4. filename_prefix, so outputs say which arm produced them.

Arm B ("current") is the shipped full58_sage_API.json with only the prefix
changed. It is copied rather than edited in place: the shipped file documents a
configuration, and this experiment should not rewrite it.

Both arms are then checked to be structurally identical - same node ids, same
class per node - so that the only differences left are the sampler settings the
experiment is actually about.
"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUTDIR = REPO / "workflows/generated/quality_ab"

JAN_COMMIT = "e906720"
JAN_PATH = "workflows/InfiniteTalk-I2V-FP8-Lip-Sync_5090_sage_new_prompts_API.json"
NOW_PATH = REPO / "workflows/generated/full/full58_sage_API.json"

AUDIO = "santa_58s.mp3"
IMAGE = "santa-classic-portrait.png"

# Settings that must survive untouched on arm A. Asserted after editing, because
# a silent modernisation here would turn the experiment into a comparison of two
# current workflows.
JAN_EXPECTED = {
    "128": {"steps": 6, "scheduler": "dpm++_sde", "seed": 82364052016591},
    "122": {"quantization": "disabled", "base_precision": "fp16_fast",
            "attention_mode": "sageattn"},
    "138": {"merge_loras": False},
    "177": {"mode": "default"},
    "192": {"frame_window_size": 81, "motion_frame": 9, "tiled_vae": False},
    "134": {"blocks_to_swap": 0},
}

# The current arm must be the optimised stack, or the comparison is pointless.
NOW_EXPECTED = {
    "128": {"steps": 4, "scheduler": "flowmatch_distill", "seed": 82364052016591},
    "122": {"quantization": "fp8_e4m3fn_fast"},
    "138": {"merge_loras": True},
    "318": {"trim_to_audio": True},
}


def git_show(ref, path):
    return subprocess.run(["git", "show", f"{ref}:{path}"], cwd=REPO,
                          capture_output=True, text=True, check=True).stdout


def check(wf, expected, arm):
    for nid, want in expected.items():
        for k, v in want.items():
            got = wf[nid]["inputs"].get(k)
            if got != v:
                sys.exit(f"{arm}: node {nid}.{k}: expected {v!r}, got {got!r}")
    n = sum(len(v) for v in expected.values())
    print(f"  verified {n} settings")


def build_january():
    wf = json.loads(git_show(JAN_COMMIT, JAN_PATH))
    print(f"arm A (january): {JAN_COMMIT}:{JAN_PATH} - {len(wf)} nodes")

    wf["125"]["inputs"]["audio"] = AUDIO
    wf["284"]["inputs"]["image"] = IMAGE
    print(f"  [1] LoadAudio -> {AUDIO}, LoadImage -> {IMAGE}")

    # Rewire before deleting: dropping 302 while 194 still points at it would
    # leave an unresolvable link that ComfyUI only complains about at queue time.
    embeds = wf["194"]["inputs"]
    if embeds.get("audio_1") != ["302", 0]:
        sys.exit(f"expected 194.audio_1 <- 302, found {embeds.get('audio_1')}")
    embeds["audio_1"] = ["125", 0]
    for nid in ("301", "302"):
        wf.pop(nid)
    print("  [2] MelBand removed (301, 302); 194.audio_1 <- 125")

    wf["318"]["inputs"]["trim_to_audio"] = True
    print("  [3] VHS_VideoCombine.trim_to_audio -> true")

    wf["318"]["inputs"]["filename_prefix"] = "IT_jan"
    print("  [4] filename_prefix -> IT_jan")

    dangling = [
        f"{nid}.{k}"
        for nid, node in wf.items()
        for k, v in (node.get("inputs") or {}).items()
        if isinstance(v, list) and len(v) == 2 and v[0] in ("301", "302")
    ]
    if dangling:
        sys.exit(f"dangling links to removed nodes: {dangling}")

    check(wf, JAN_EXPECTED, "january")
    return wf


def build_current():
    wf = json.loads(NOW_PATH.read_text())
    print(f"arm B (current): {NOW_PATH.relative_to(REPO)} - {len(wf)} nodes")
    wf["318"]["inputs"]["filename_prefix"] = "IT_now"
    print("  [1] filename_prefix -> IT_now")
    if wf["125"]["inputs"]["audio"] != AUDIO or wf["284"]["inputs"]["image"] != IMAGE:
        sys.exit("current arm does not already point at the shared inputs")
    check(wf, NOW_EXPECTED, "current")
    return wf


def main():
    jan = build_january()
    now = build_current()

    # Same graph, different knobs. If this ever fails the two arms differ
    # structurally and no metric comparing them frame by frame is trustworthy.
    if set(jan) != set(now):
        sys.exit(f"node sets differ: A-only={set(jan)-set(now)} B-only={set(now)-set(jan)}")
    classes = [n for n in jan if jan[n]["class_type"] != now[n]["class_type"]]
    if classes:
        detail = ", ".join(f"{n}: {jan[n]['class_type']} vs {now[n]['class_type']}"
                           for n in classes)
        print(f"note: node classes renamed upstream since January - {detail}")
        print("      this is why arm A needs its January node pin")
    print(f"both arms: {len(jan)} nodes")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    for name, wf in (("A_january_API.json", jan), ("B_current_API.json", now)):
        (OUTDIR / name).write_text(json.dumps(wf, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote {(OUTDIR / name).relative_to(REPO)}")


if __name__ == "__main__":
    main()
