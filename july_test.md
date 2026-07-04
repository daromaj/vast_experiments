# July 2026 InfiniteTalk optimization test

Goal: beat the reliable ~12 min / 60s @ 480p baseline (`InfiniteTalk-I2V-FP8-Lip-Sync_5090_sage_new_prompts.json`) on a single RTX 5090, without losing lip-sync/motion quality.

## What changed vs baseline

| Setting | Baseline | `IT_5090_july2026_4step` | `IT_5090_july2026_5step` |
|---|---|---|---|
| steps | 6 | 4 | 5 |
| scheduler | dpm++_sde | flowmatch_distill | dpm++_sde |
| quantization | disabled | fp8_e4m3fn_fast | fp8_e4m3fn_fast |
| merge_loras | false | true | true |
| compile mode | default | max-autotune-no-cudagraphs | max-autotune-no-cudagraphs |

Unchanged: 832×480, 81f windows, motion_frame 9, cfg 1.0, shift 11, sageattn, blocks_to_swap 0.

Also: MelBandRoFormer vocal-separation nodes are **bypassed** (LoadAudio feeds MultiTalkWav2VecEmbeds directly). Only valid when the input audio is already clean voice — if you feed a song, re-enable them (set both MelBand nodes back to mode 0 and rewire audio_1 through the sampler). Negligible speed gain; removes one preprocessing pass + model load.

Why: the baseline runs 6 steps on a LoRA distilled for 4 (`lightx2v cfg_step_distill`), never engages the fp8 matmul path (needs `_fast` + merged LoRA), and compiles in `default` mode. These are the three additive, independent levers.

## Prep

1. Provision a 5090 with `povision_fp8.sh` (already lists both new workflows — they land in `ComfyUI/user/default/workflows/`).
2. Use the **same input image + same audio** for every run so timings/quality are comparable. Put them in `input/`.
3. Fix the seed (already fixed in the workflow) so quality diffs are real, not noise.

## Test procedure

Run in this order. Test one workflow at a time.

### Step 0 — re-baseline (optional but do it once)
Run `InfiniteTalk-I2V-FP8-Lip-Sync_5090_sage_new_prompts.json` on THIS instance with your standard 60s audio. Record wall-clock of the **2nd** generation (first one may compile). This is your reference number on current hardware/driver/node versions — your old 12 min figure is stale.

### Step 1 — `IT_5090_july2026_4step` (aggressive)
1. Load it, queue one 60s generation. **Ignore this run's time** — max-autotune spends a few extra minutes compiling on first run per shape.
2. Queue a **second** identical generation. This is the number that counts. Record wall-clock.
3. Inspect output quality (see checklist below).

### Step 2 — `IT_5090_july2026_5step` (fallback)
Same procedure. Only worth running if 4step shows quality problems, or to see the speed/quality midpoint.

## What to confirm in the ComfyUI log

- `FP8 matmul enabled` → fp8_fast is actually engaged. If you don't see it, the merge/quant combo didn't take.
- No flood of `torch._dynamo ... graph break` / recompile warnings during sampling. A few at startup are fine; continuous ones mean compile is fighting the fp8 lambda path — note it, we may drop that variant's compile mode back to `default`.
- Note the one-time compile duration separately from generation time.

## Quality checklist (compare against baseline output)

- [ ] Lip-sync stays locked to audio for the full 60s (worst case: drift near the end)
- [ ] No jump-cuts / motion discontinuity at window seams (~every 3s)
- [ ] No new color shift or washed-out look vs baseline
- [ ] Face identity preserved (fp8 matmul is lossier than fp16 — check it didn't melt the face)

## Record results (fill in)

| Workflow | Compile time (1st run) | Gen time (2nd run) | vs baseline | Quality verdict |
|---|---|---|---|---|
| baseline (re-run) | n/a | | — | |
| july2026_4step | | | | |
| july2026_5step | | | | |

## Decision

- 4step passes quality → make it the new default, retire the 6-step baseline.
- 4step fails, 5step passes → use 5step.
- Both fail quality → isolate which lever broke it: re-test with fp8_fast OFF (quant back to `disabled`, merge_loras `false`) at 4/5 steps to see if fp8 or the step cut is the culprit.

## Reminder on the ceiling

None of these reach the API provider's ~6–7 min. That's near-certainly multi-GPU Ulysses sequence parallelism (officially supported: `torchrun --ulysses_size=N --dit_fsdp`), which the single-GPU kijai wrapper can't do. To actually match the API you'd leave ComfyUI and run `generate_infinitetalk.py` under torchrun on a multi-GPU box. Separate experiment.
