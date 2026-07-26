#!/usr/bin/env python3
"""
Apply the three zero-code, zero-quality-risk wins from the node source audit
(notes/node_optimization_audit.md) to an API-format workflow.

All three are things the workflow was doing wrong against the node's own
defaults or against an input that was simply never wired up. None of them change
what the model computes, so output should be bit-identical at a fixed seed - the
only thing that moves is wall-clock.

R1  241.use_disk_cache false -> true
    The node default is true (nodes.py:200). With it on and the prompt unchanged,
    get_cached_text_embeds hits and the node returns before
    LoadWanVideoT5TextEncoder().loadmodel() ever runs (nodes.py:257-266), so
    umt5-xxl (~11GB) is never read off disk. Description says it outright:
    "If prompts have been cached before T5 is not loaded at all" (nodes.py:213-215).

R2  wire 177 (WanVideoTorchCompileSettings) -> 129.compile_args
    WanVideoVAELoader has an optional compile_args input (nodes_model_loading.py
    :1888) that torch.compiles vae.model.decoder (:1925-1926). Nothing was
    connected. The decoder is invoked once per latent frame in a Python loop
    (wan_video_vae.py:1143-1153) - 21 calls per window, 441 over a 58s clip -
    so it is launch-overhead-bound and exactly what inductor fusion is for.

R5  128.force_offload true -> false
    Post-run only (multitalk_loop.py:561-563), so it costs nothing inside a run,
    but it pushes ~16GB of transformer to CPU at the end and the next queued
    generation pays to load it back. Peak was 24.9 of 31.36 GiB, so there is room
    to just leave the weights resident.
"""
import argparse
import json

N_TEXT_ENCODE = "241"
N_VAE_LOADER = "129"
N_COMPILE = "177"
N_SAMPLER = "128"


def apply_wins(wf):
    changed = []

    enc = wf[N_TEXT_ENCODE]["inputs"]
    if enc.get("use_disk_cache") is not True:
        enc["use_disk_cache"] = True
        changed.append("R1 241.use_disk_cache -> true")

    vae = wf[N_VAE_LOADER]["inputs"]
    if vae.get("compile_args") != [N_COMPILE, 0]:
        vae["compile_args"] = [N_COMPILE, 0]
        changed.append("R2 129.compile_args <- 177")

    smp = wf[N_SAMPLER]["inputs"]
    if smp.get("force_offload") is not False:
        smp["force_offload"] = False
        changed.append("R5 128.force_offload -> false")

    return changed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src")
    ap.add_argument("dest")
    args = ap.parse_args()

    wf = json.load(open(args.src))
    changed = apply_wins(wf)
    with open(args.dest, "w") as fh:
        json.dump(wf, fh, indent=1)

    for line in changed:
        print(f"  {line}")
    print(f"{len(changed)} change(s) -> {args.dest}")


if __name__ == "__main__":
    main()
