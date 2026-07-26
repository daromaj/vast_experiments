#!/usr/bin/env python3
"""
Generate API-format workflows for the july_test.md candidates.

The july2026 workflows only exist in UI format, which ComfyUI's /prompt endpoint
cannot execute. Rather than write a general UI->API converter (which needs every
node's input schema and is easy to get subtly wrong), this derives them from the
existing baseline API workflow by applying exactly the deltas that
`diff_workflows.py` reports between the UI baseline and each UI candidate.

Deltas confirmed against the UI files:

  4step: quantization fp8_e4m3fn_fast, steps 6->4, scheduler flowmatch_distill,
         merge_loras True, compile mode max-autotune-no-cudagraphs, MelBand bypassed
  5step: same but steps 6->5 and scheduler left at dpm++_sde

Bypassing MelBand (mode 4 in the UI) means dropping the two nodes and rewiring
MultiTalkWav2VecEmbeds.audio_1 straight to LoadAudio - valid only because the
input audio is already clean voice. See july_test.md.
"""
import argparse
import copy
import json
import os

BASELINE = "workflows/InfiniteTalk-I2V-FP8-Lip-Sync_5090_sage_new_prompts_API.json"

# Node ids are stable across these workflows (verified: identical node sets).
N_MODEL_LOADER = "122"
N_SAMPLER = "128"
N_LORA = "138"
N_COMPILE = "177"
N_MELBAND_LOADER = "301"
N_MELBAND_SAMPLER = "302"
N_WAV2VEC_EMBEDS = "194"
N_LOAD_AUDIO = "125"
N_LOAD_IMAGE = "284"


def bypass_melband(wf):
    """Drop the vocal-separation pair and feed LoadAudio straight to the embeds."""
    wf[N_WAV2VEC_EMBEDS]["inputs"]["audio_1"] = [N_LOAD_AUDIO, 0]
    wf.pop(N_MELBAND_LOADER, None)
    wf.pop(N_MELBAND_SAMPLER, None)


def make_variant(base, steps, scheduler):
    wf = copy.deepcopy(base)
    wf[N_MODEL_LOADER]["inputs"]["quantization"] = "fp8_e4m3fn_fast"
    wf[N_SAMPLER]["inputs"]["steps"] = steps
    wf[N_SAMPLER]["inputs"]["scheduler"] = scheduler
    wf[N_LORA]["inputs"]["merge_loras"] = True
    wf[N_COMPILE]["inputs"]["mode"] = "max-autotune-no-cudagraphs"
    bypass_melband(wf)
    return wf


N_TEXT_ENCODE = "241"


def set_inputs(wf, image, audio):
    wf[N_LOAD_IMAGE]["inputs"]["image"] = image
    wf[N_LOAD_AUDIO]["inputs"]["audio"] = audio


def set_text_encode(wf, device, quantization, disk_cache):
    """
    The shipped workflow runs umt5-xxl-enc-bf16 (10.58GB) on 'gpu' with no disk
    cache, and that configuration is known to work on a 32GB 5090.

    An OOM here is almost certainly NOT the encoder. InfiniteTalk samples in
    81-frame windows but retains decoded pixel frames to seed each next window's
    motion frames, so VRAM scales with TOTAL clip length. With blocks_to_swap=0
    the 15.8GB diffusion model plus 2.5GB InfiniteTalk stay fully resident, which
    leaves little headroom: a 58s clip (1455 frames) OOMs in WanVideoSampler
    where an 8s clip fits comfortably. Shorten the clip or raise blocks_to_swap
    rather than reaching for these knobs.

    They exist anyway because they are genuinely useful: fp8_e4m3fn drops the
    encoder to ~5.3GB, and disk_cache lets reruns skip encoding entirely. Both
    apply identically to every variant, so speed comparisons stay valid.
    """
    n = wf[N_TEXT_ENCODE]["inputs"]
    n["device"] = device
    n["quantization"] = quantization
    n["use_disk_cache"] = disk_cache


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", default="santa-classic-portrait.png")
    ap.add_argument("--audio", default="santa_58s.mp3")
    ap.add_argument("--outdir", default="workflows/generated")
    # Defaults reproduce the configuration that is known to have worked. The
    # OOM chased earlier was caused by feeding a 58s clip, not by the encoder -
    # see set_text_encode. Do not change these to "fix" an OOM; shorten the clip
    # or raise blocks_to_swap instead.
    ap.add_argument("--text-encode-device", default="gpu", choices=["cpu", "gpu"])
    ap.add_argument("--text-encode-quantization", default="disabled",
                    choices=["disabled", "fp8_e4m3fn"],
                    help="disabled=bf16 10.58GB (as shipped); fp8_e4m3fn=~5.3GB")
    ap.add_argument("--disk-cache", action="store_true",
                    help="cache prompt embeddings to disk so reruns skip encoding")
    args = ap.parse_args()

    base = json.load(open(BASELINE))
    os.makedirs(args.outdir, exist_ok=True)

    variants = {
        # Baseline keeps MelBand active and every original setting - only the
        # input filenames change, so it is a true reference point.
        "baseline": copy.deepcopy(base),
        "july4step": make_variant(base, 4, "flowmatch_distill"),
        "july5step": make_variant(base, 5, "dpm++_sde"),
    }

    for name, wf in variants.items():
        set_inputs(wf, args.image, args.audio)
        set_text_encode(wf, args.text_encode_device,
                        args.text_encode_quantization, args.disk_cache)
        path = os.path.join(args.outdir, f"IT_5090_{name}_API.json")
        with open(path, "w") as fh:
            json.dump(wf, fh, indent=1)
        s = wf[N_SAMPLER]["inputs"]
        m = wf[N_MODEL_LOADER]["inputs"]
        print(
            f"{name:<10} steps={s['steps']} sched={s['scheduler']:<18} "
            f"quant={m['quantization']:<16} merge_loras={wf[N_LORA]['inputs']['merge_loras']} "
            f"compile={wf[N_COMPILE]['inputs']['mode']:<26} "
            f"melband={'bypassed' if N_MELBAND_SAMPLER not in wf else 'active'} "
            f"txtenc={wf[N_TEXT_ENCODE]['inputs']['device']}/"
            f"{wf[N_TEXT_ENCODE]['inputs']['quantization']} "
            f"nodes={len(wf)}"
        )
        print(f"           -> {path}")


if __name__ == "__main__":
    main()
