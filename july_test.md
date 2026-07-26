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

## Results — 2026-07-26, RTX 5090 (31.36 GiB), vast.ai 45930851

**Read this first: the OOM was a SageAttention wheel built for the wrong GPU.**

Every workflow here OOMed in `WanVideoSampler` at ~29.3 GiB, and switching node
122 to `sdpa` was what made them complete — which made `sageattn` look like the
culprit. It wasn't. The wheel cached at
`python/sage/torch2.10.0-cu128-sm_120/` was dated 2025-12-13 and had been built
for Ada/Ampere, so on a 5090 SageAttention **silently fell back** and used ≥6 GB
more VRAM than SDPA. `sage_abi_probe.py` passed it because a fallback still
returns *correct* output (cosine 0.9993 vs SDPA).

Rebuilt from source on the 5090 itself with `TORCH_CUDA_ARCH_LIST=12.0` (nvcc:
`396 entry functions for 'sm_120a'`), sage is not just fine — it is the single
biggest speed lever available. That wheel is now what's committed.

Do not try to identify a sage build by filename: 2.2.0 always produces
`_qattn_sm80` / `_qattn_sm89` extensions no matter what it was compiled for,
because those are the kernel *sources*, not the targets. The probe now checks
behaviourally, comparing peak VRAM against SDPA.

Measured with `scripts/run_july_tests.py` (2 runs each; run 1 discarded as
compile warmup). Input: 8 s clip @ 480×832, `blocks_to_swap=0`.

| Config | Attention | Steps | Scheduler | Tiled VAE | Compile | Warmup | **Gen time** | vs baseline |
|---|---|---|---|---|---|---|---|---|
| **s5_sage_untiled** | sageattn | 4 | flowmatch_distill | no | max-autotune | 120.1 s | **87.7 s** | **2.41× faster** |
| s1_4step_untiled | sdpa | 4 | flowmatch_distill | no | max-autotune | 150.3 s | 117.8 s | 1.80× |
| s0_4step_tiled | sdpa | 4 | flowmatch_distill | yes | max-autotune | 171.7 s | 141.2 s | 1.50× |
| s3_4step_nocomp | sdpa | 4 | flowmatch_distill | yes | default | 174.2 s | 141.4 s | 1.50× |
| s2_5step_tiled | sdpa | 5 | dpm++_sde | yes | max-autotune | 195.2 s | 162.7 s | 1.30× |
| s4_baseline_sdpa | sdpa | 6 | dpm++_sde | yes | default | 231.7 s | 211.5 s | — |

Conclusions:

- **SageAttention is worth 30 s** (117.8 → 87.7) at otherwise identical
  settings — provided the wheel matches the host arch. `scripts/attention_bench.py`
  measures the kernel in isolation at the real WanVideo shape (40 heads ×
  32,760 tokens × 128 dim): **sage 39.1 ms vs sdpa 106.9 ms**.
- **`tiled_vae` costs ~23 s** (141.2 → 117.8). It was only ever an OOM
  workaround; there is headroom without it. Turn it off.
- **`max-autotune-no-cudagraphs` buys nothing** on sdpa: 141.2 s vs 141.4 s for
  plain `default`, while adding ~33 s to the first run. Net loss unless you
  generate many clips per ComfyUI session.
- 4 steps + `flowmatch_distill` beats 5 steps by 45 s, as designed.

The four shipped `IT_{4090,5090}_july2026_{4,5}step.json` workflows already
carry this configuration (sageattn, `tiled_vae=false`, 81-frame window,
`max-autotune-no-cudagraphs`); nothing in them needed changing once the wheel
was fixed.

Quality has NOT been assessed yet — these are speed numbers only. Compare the
pulled videos before adopting 4-step as the default.

Outputs pulled to `output/vast_45930851/` (gitignored, kept locally). The second
file of each pair is the measured run:

| File | Config |
|---|---|
| `InfiniteTalk_00003/00004` | s0_4step_tiled |
| `InfiniteTalk_00005/00006` | **s1_4step_untiled** (fastest) |
| `InfiniteTalk_00007/00008` | s2_5step_tiled |
| `InfiniteTalk_00009/00010` | s3_4step_nocomp |
| `InfiniteTalk_00011/00012` | s4_baseline_sdpa (6-step reference) |
| `InfiniteTalk_00013/00014` | **s5_sage_untiled** (fastest) |

Compare `00014` (4-step sage) against `00012` (6-step baseline) — that is the
quality question that decides whether the 2.41× is free. `00006` (4-step sdpa)
isolates the attention backend from the step count.

### Measurement trap worth keeping

Re-queueing a byte-identical workflow does not regenerate anything: ComfyUI
caches on node inputs and returns the previous result. This showed up as a
"3.0 s generation" that emitted the same filename as the 171.5 s run before it.
`run_july_tests.py` now bumps every `seed`/`noise_seed` per run, which forces
real resampling while leaving the torch.compile cache warm — which is exactly
what run 2 is meant to measure.

## Decision

- 4step passes quality → make it the new default, retire the 6-step baseline.
- 4step fails, 5step passes → use 5step.
- Both fail quality → isolate which lever broke it: re-test with fp8_fast OFF (quant back to `disabled`, merge_loras `false`) at 4/5 steps to see if fp8 or the step cut is the culprit.

## 4090 variants (24 GB)

`IT_4090_july2026_4step.json` / `_5step.json` are identical to the 5090 variants **except block swap** — the only setting that must change for Ada/24 GB. fp8 `_scaled_mm`, SageAttention 2.x, fp16_fast and max-autotune all work on a 4090; VRAM is the sole constraint.

| Setting | 5090 (32 GB) | 4090 (24 GB) |
|---|---|---|
| blocks_to_swap | 0 | 20 |
| prefetch_blocks | 0 | 1 |

`blocks_to_swap=20` (of the 14B model's 40 blocks) is the node default and a **safe starting point** — not a tuned value. `prefetch_blocks=1` overlaps the CPU→GPU block transfer with compute to claw back most of the swap penalty.

**Tune it on the host:**
- Watch `nvidia-smi` peak VRAM during a run. If it OOMs, raise `blocks_to_swap` (try 24–28).
- If there's several GB of headroom to spare, **lower** it (try 12–16) — fewer swapped blocks = faster. Every block you keep resident is time saved.
- Expect the 4090 to be slower than the 5090 regardless: 24 GB can't hold the whole model, so some swap traffic is unavoidable. These variants make it *run* and be as fast as 24 GB allows, not match the 5090.

## Reminder on the ceiling

None of these reach the API provider's ~6–7 min. That's near-certainly multi-GPU Ulysses sequence parallelism (officially supported: `torchrun --ulysses_size=N --dit_fsdp`), which the single-GPU kijai wrapper can't do. To actually match the API you'd leave ComfyUI and run `generate_infinitetalk.py` under torchrun on a multi-GPU box. Separate experiment.
