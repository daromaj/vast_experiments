#!/usr/bin/env python3
"""
Emit candidate workflows for a memory/speed sweep on a 32GB RTX 5090.

Why this exists: the shipped workflow uses base_precision=fp16_fast with
quantization=disabled, which makes WanVideoModelLoader materialise the 14B
transformer at fp16 (~28.6GB) and, with load_device=main_device, pin it in VRAM
(nodes_model_loading.py:1810 skips the offload entirely). Against a 31.36GiB
limit that leaves nothing for sampling, and it OOMs in WanVideoSampler.

Verified NOT to be the cause, so don't re-test these:
  - the text encoder (fp8 vs bf16 moved peak by 160MB)
  - the workflow itself (byte-identical to the 2026-01-24 known-good revision
    apart from the wav2vec model name)
  - WanVideoWrapper drift (pinning back to 339e0fe still OOMs)

So the levers that remain are the ones that actually change resident bytes:
quantization (fp8 keeps weights at ~16GB instead of dequantising to fp16),
blocks_to_swap (offload N of 40 transformer blocks), and frame_window_size
(attention is over frame_window_size/4 * 60 * 104 tokens, so it drives peak
activation memory during sampling).

Candidates are ordered most-frugal-first: the point is to find something that
completes at all, then walk back toward speed.
"""
import argparse
import copy
import json
import os

BASE = "workflows/InfiniteTalk-I2V-FP8-Lip-Sync_5090_sage_new_prompts_API.json"

N_MODEL_LOADER = "122"
N_SAMPLER = "128"
N_LORA = "138"
N_COMPILE = "177"
N_BLOCK_SWAP = "134"
N_MELBAND_LOADER = "301"
N_MELBAND_SAMPLER = "302"
N_WAV2VEC_EMBEDS = "194"
N_LOAD_AUDIO = "125"
N_LOAD_IMAGE = "284"
N_I2V_MULTITALK = "192"

# The weight-side levers were the wrong axis: an audit of the loader showed
# quantization="disabled" AUTODETECTS the checkpoint's fp8 (nodes_model_loading
# .py:1187-1201, :1717-1722), so the transformer is already ~15.5GiB, not the
# ~28.6GiB fp16 first assumed. Static weight math accounts for only ~19-21GiB of
# the observed 29.3GiB.
#
# The unaccounted ~10GiB is on the ACTIVATION side, inside node 128 - which is
# exactly where every OOM lands. Node 192 owns the two knobs for that, and
# neither had ever been tried:
#   force_offload - offloads the whole transformer between windows
#                   (multitalk/nodes.py tooltip: "enable if you encounter
#                   memory issues"; multitalk_loop.py:462-464)
#   tiled_vae     - VAE encode/decode of an 81-frame 480x832 window in one shot
#                   is a large transient spike; tiling caps it
#
# Ordered most-frugal-first: find something that completes, then walk back
# toward speed by relaxing one knob at a time.
# SOLVED 2026-07-26: attention_mode=sageattn was the OOM. Every sage run died at
# ~29GiB in WanVideoSampler; switching node 122 to sdpa completed in 187.6s on
# the same instance, same clip, everything else equal. sdpa is also the node's
# own default (nodes_model_loading.py:1022) - the workflow had overridden it.
# The image ships only `sageattention` (no flash_attn, no sageattn3), so sdpa is
# the backend to use here.
#
# These candidates now sweep SPEED, not survival. Two runs each: the first pays
# max-autotune compile cost, only the second is a real generation time.
# name, quant, swap, window, steps, scheduler, compile, force_offload, tiled_vae, attention
CANDIDATES = [
    ("s0_4step_tiled",   "fp8_e4m3fn_fast", 0, 81, 4, "flowmatch_distill", "max-autotune-no-cudagraphs", False, True,  "sdpa"),
    ("s1_4step_untiled", "fp8_e4m3fn_fast", 0, 81, 4, "flowmatch_distill", "max-autotune-no-cudagraphs", False, False, "sdpa"),
    ("s2_5step_tiled",   "fp8_e4m3fn_fast", 0, 81, 5, "dpm++_sde",         "max-autotune-no-cudagraphs", False, True,  "sdpa"),
    ("s3_4step_nocomp",  "fp8_e4m3fn_fast", 0, 81, 4, "flowmatch_distill", "default",                    False, True,  "sdpa"),
    ("s4_baseline_sdpa", "disabled",        0, 81, 6, "dpm++_sde",         "default",                    False, True,  "sdpa"),
]


def bypass_melband(wf):
    wf[N_WAV2VEC_EMBEDS]["inputs"]["audio_1"] = [N_LOAD_AUDIO, 0]
    wf.pop(N_MELBAND_LOADER, None)
    wf.pop(N_MELBAND_SAMPLER, None)


def build(base, quant, swap, window, steps, scheduler, compile_mode,
          force_offload, tiled_vae, attention, image, audio):
    wf = copy.deepcopy(base)
    m = wf[N_MODEL_LOADER]["inputs"]
    m["quantization"] = quant
    m["attention_mode"] = attention
    # Keep main_device even when swapping. With block_swap_args present the
    # loader already picks offload_device per block at load time
    # (nodes_model_loading.py:912-920), so swapping works here - while
    # load_device=offload_device would put ALL fp8 params on CPU, which
    # segfaults in apply_lora's scale-weight branch (utils.py:343, observed:
    # SIGSEGV taking the whole ComfyUI process down).
    m["load_device"] = "main_device"
    wf[N_BLOCK_SWAP]["inputs"]["blocks_to_swap"] = swap
    wf[N_SAMPLER]["inputs"]["steps"] = steps
    wf[N_SAMPLER]["inputs"]["scheduler"] = scheduler
    wf[N_COMPILE]["inputs"]["mode"] = compile_mode
    # merge_loras=True bakes the LoRA into the fp8 weights, removing the separate
    # unmerged-diff buffers (~0.7GiB) and the per-forward add.
    wf[N_LORA]["inputs"]["merge_loras"] = quant != "disabled"
    wf[N_I2V_MULTITALK]["inputs"]["frame_window_size"] = window
    wf[N_I2V_MULTITALK]["inputs"]["force_offload"] = force_offload
    wf[N_I2V_MULTITALK]["inputs"]["tiled_vae"] = tiled_vae
    wf[N_LOAD_IMAGE]["inputs"]["image"] = image
    wf[N_LOAD_AUDIO]["inputs"]["audio"] = audio
    if quant != "disabled":
        bypass_melband(wf)
    return wf


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", default="santa-classic-portrait.png")
    ap.add_argument("--audio", default="santa_8s.mp3")
    ap.add_argument("--outdir", default="workflows/generated/sweep")
    args = ap.parse_args()

    base = json.load(open(BASE))
    os.makedirs(args.outdir, exist_ok=True)

    for name, quant, swap, window, steps, sched, cmode, foff, tvae, attn in CANDIDATES:
        wf = build(base, quant, swap, window, steps, sched, cmode,
                   foff, tvae, attn, args.image, args.audio)
        path = os.path.join(args.outdir, f"{name}_API.json")
        with open(path, "w") as fh:
            json.dump(wf, fh, indent=1)
        print(f"{name:<18} quant={quant:<16} swap={swap:<3} window={window:<3} "
              f"steps={steps} sched={sched:<18} tiled_vae={tvae} attn={attn}")
        print(f"    -> {path}")


if __name__ == "__main__":
    main()
