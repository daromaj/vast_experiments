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
| **s9_sage_patched** | sageattn | 4 | flowmatch_distill | no | max-autotune | 193.3 s | **72.3 s** | **2.93× faster** |
| s8_sage_quickwins | sageattn | 4 | flowmatch_distill | no | max-autotune | 646.5 s | 81.7 s | 2.59× |
| s5_sage_untiled | sageattn | 4 | flowmatch_distill | no | max-autotune | 120.1 s | 87.7 s | 2.41× |
| s1_4step_untiled | sdpa | 4 | flowmatch_distill | no | max-autotune | 150.3 s | 117.8 s | 1.80× |
| s0_4step_tiled | sdpa | 4 | flowmatch_distill | yes | max-autotune | 171.7 s | 141.2 s | 1.50× |
| s3_4step_nocomp | sdpa | 4 | flowmatch_distill | yes | default | 174.2 s | 141.4 s | 1.50× |
| s2_5step_tiled | sdpa | 5 | dpm++_sde | yes | max-autotune | 195.2 s | 162.7 s | 1.30× |
| s4_baseline_sdpa | sdpa | 6 | dpm++_sde | yes | default | 231.7 s | 211.5 s | — |

### The two rounds of optimization on top of sage

`s8` and `s9` come from a source audit of ComfyUI-WanVideoWrapper
(`notes/node_optimization_audit.md`, cites checked against the installed copy,
which was byte-identical to the audited HEAD).

**s8 — three changes, no code** (`scripts/apply_quick_wins.py`, and
`apply_quick_wins_gui.py` for the shipped GUI-format workflows):

| | Change | Evidence it worked |
|---|---|---|
| R1 | `WanVideoTextEncodeCached.use_disk_cache` false → **true** | log shows `Loading prompt embeds from cache: .../text_embed_cache/4dc8d4ae….pt` — umt5-xxl (~11 GB) never loads |
| R2 | wire `WanVideoTorchCompileSettings` → `WanVideoVAELoader.compile_args` | compiles `vae.model.decoder` (`nodes_model_loading.py:1925-1926`), which runs 21 launch-bound calls per window |
| R5 | `WanVideoSampler.force_offload` true → **false** | post-run only; stops pushing ~16 GB to CPU between queued runs |

The node default for `use_disk_cache` is `true` (`nodes.py:200`) — the workflow
had overridden it to `false`, which is why the T5 was reloading every run.

Note R2's price: **the first run after a restart costs 646.5 s** while inductor
autotunes the decoder convs. It is amortized from run 2 onward and cached on
disk, but on a one-shot rental that generates a single clip, R2 is a net loss.

**s9 — two code patches** (`scripts/patch_multitalk_loop.py`, env-gated so the
same file runs both arms of the A/B):

| | Change | Why it is safe |
|---|---|---|
| R3 | cache the 81-frame `y` VAE encode | With one start image, `cond_` is always `cond_image` and the other 80 frames are zeros, so `vae.encode` returns a bit-identical tensor every window. Cache is keyed on a fingerprint of the real tensor, not on that assumption, so a changing input simply misses. Log confirms `MISS` then `HIT`. |
| R6 | skip the per-window `soft_empty_cache()`×2 + `gc.collect()` | No numerical effect. These hold peak VRAM down; there was 6.5 GiB of headroom. Put back first if you raise `frame_window_size`. |

Together: 81.7 → **72.3 s**.

Conclusions:

- **SageAttention is worth 30 s** (117.8 → 87.7) at otherwise identical
  settings — provided the wheel matches the host arch. `scripts/attention_bench.py`
  measures the kernel in isolation at the real WanVideo shape (40 heads ×
  32,760 tokens × 128 dim): **sage 39.1 ms vs sdpa 106.9 ms**.
- **`tiled_vae` costs ~23 s** (141.2 → 117.8). It was only ever an OOM
  workaround; there is headroom without it. Turn it off.
- **`max-autotune-no-cudagraphs` buys nothing** on sdpa: 141.2 s vs 141.4 s for
  plain `default`, while adding ~33 s to the first run. Net loss unless you
  generate many clips per ComfyUI session. (On *sage* it is a clear win, and it
  pays back within a single 58 s clip — see the one-shot section below. The
  ~33 s first-run figure is an 8 s-clip number and does not scale; measured on a
  58 s clip the cold penalty is ~75 s.)
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

`prefetch_blocks=1` overlaps the CPU→GPU block transfer with compute to claw back most of the swap penalty.

### Results — 2026-07-27, RTX 4090 (24,564 MiB), vast.ai 45944385

`blocks_to_swap=20` was the node default and an untuned guess. It is now **measured and kept**. 8 s clip, run 2 of 2, sage + 4-step + quick wins, untiled:

| swap | warm | peak reserved | |
|---|---|---|---|
| 2 | — | 24,068 MiB (pinned) | **OOM** |
| 4 | 254.9 s | 24,080 MiB (pinned) | |
| 6 | 255.8 s | 24,076 MiB (pinned) | |
| 12 | 108.9 s | 21,782 MiB | |
| **16** | **105.2 s** | 20,562 MiB | fastest |
| 20 | 106.9 s | 20,562 MiB | **shipped** |
| 28 | 114.5 s | 20,562 MiB | |

Full 58 s clip at swap=16: **816.3 s (13 m 36 s)**, peak 20,626 MiB, versus 9 m 50 s on the 5090 — **1.38x slower**.

**The advice this section used to give was backwards.** It said to lower `blocks_to_swap` when there is headroom, because "every block you keep resident is time saved". The opposite is true near the ceiling:

- torch reports a **device limit of 23.52 GiB**, not the 23.99 GiB the nameplate implies — ~480 MB goes to driver and context. Comparing *total* VRAM is what made the 4090 look ~376 MB short; against *usable* memory the gap is ~0.9 GiB.
- swap=4 and swap=6 both sit pinned at that limit and run **2.4x slower** than swap=16. Pinned against the ceiling the caching allocator thrashes on synchronous frees and `cudaMalloc` retries, which costs far more than the PCIe traffic of a dozen extra blocks.
- The curve is U-shaped with a flat floor from 12 to 20, turning up only at 28 where transfer cost finally dominates.

So the tuning rule is **not** "shed the minimum that avoids OOM" — that lands squarely in the 4-6 thrash zone, the slowest working setting on the card. Target real headroom instead: ~20,500 MiB reserved against a 23.52 GiB limit. 16 and 20 are within noise of each other and report identical peaks, so the shipped 20 stays.

This also explains the pre-fix figures in `README.md` (`swap 5 = 238 s` beating `swap 20 = 314 s`), which are the reverse ordering. Those were measured with the mis-built SageAttention wheel; correcting the wheel flips the result.

Caveat on the peak column: it is *reserved*, read from `nvidia-smi`, not required. The caching allocator does not hand memory back, so a pinned variant reports the device limit whatever its true working set. It separates "had headroom" from "had none" and nothing finer — `torch.cuda.max_memory_allocated` inside the node would be needed for anything sharper.

## One-shot rental — 2026-07-27, RTX 5090, vast.ai 45967149

Every figure above is per-render on a box that was already provisioned and, from
run 2 onward, already warm. That is the wrong measurement for the way this is
actually used: rent a box, make one video, destroy it. Then provisioning and the
cold inductor cache are paid **per video**, not amortized across a sweep.

Measured end to end with `scripts/e2e_oneshot.sh`, which stamps each boundary as
it happens (`notes/run_results/e2e_oneshot/phases.tsv`) rather than reconstructing
it from logs afterwards:

| Phase | Duration |
|---|---|
| create → ssh reachable | 0 m 49 s |
| provisioning | 6 m 27 s |
| upload workflow + assets | 0 m 04 s |
| **render 58 s clip, cold cache** | **11 m 06 s** |
| download outputs | 0 m 02 s |
| destroy | 0 m 01 s |
| **total billed** | **18 m 29 s** |

Output verified: 60.84 s, 480x832, 1521 frames, audio track present, 8.6 MB.
Peak VRAM 26,474 MiB — above the ~24.4 GiB seen in the sweep because R6
(`WANOPT_KEEP_CACHE_WARM=1`) skips the per-window cache purge, so the allocator
keeps more. Still comfortable on 32 GB.

**Only 60% of the wall clock is generation.** The remaining 7 m 20 s is boot and
provisioning.

### The cold-compile penalty is ~75 s, not ~33 s

The render took 664.8 s against 589.9 s for the same workflow on a warm inductor
cache. The ~33 s cold/warm delta measured on an 8 s clip **does not carry over** —
a longer clip exercises more graph variants, so more gets compiled. Do not
extrapolate warmup cost from short clips.

It still pays for itself on a single 58 s video: without `torch.compile` the loss
is roughly 10 s per window across 21 windows, well over the 75 s it costs. The
warmup that genuinely does *not* pay back one-shot remains **R2**, the VAE decoder
compile, at 646.5 s — which is why `full58_sage_API.json` (the s5 lineage) is the
right workflow for this mode and s8 is not.

Provisioning was 5 m 59 s, of which **4 m 01 s was `apt install`** — host
variance, not a regression. SageAttention cost 12 s: the cached `sm_120` wheel
passed its probe and the source build was cancelled. The 34 GB of models pulled
in 1 m 38 s on a 7,944 Mb/s link. Both WANOPT flags were set automatically, so
this run had the s9 configuration.

### Host selection for one-shot is not what the search optimizes

This run cost **$0.148 rental + $0.088 egress ≈ $0.236**. Egress was **37% of the
bill**.

`interactive_search_vastai.py` filters on `inet_down >= 5000` Mb/s. For a sweep
that is right — the download happens once and then you render all day. For
one-shot it is backwards: the 34 GB is re-pulled for every video, so $/GB
dominates, and every host clearing that speed filter charged $2.60–$10.00/TB.

Ranking the same GPUs by full one-shot cost instead
(`scripts/search_cheap_egress.py`, which charges each offer for its own download
time so a slow link cannot look free):

| machine | $/hr | $/TB | down Mb/s | one-shot |
|---|---|---|---|---|
| 32637 (Alberta) | 0.482 | **0.33** | 1,834 | **$0.142** |
| 54134 (Nebraska) | 0.534 | **0.00** | 1,699 | $0.147 |
| 108568 (Malaysia) | 0.401 | 1.30 | 1,377 | $0.159 |
| 141151 (used here) | 0.481 | 2.60 | 7,944 | $0.204 |

A 4x slower link adds ~1 min of download, worth ~$0.015 of rental; the cheaper
egress saves ~$0.078. **Cheap egress beats a fast link at this transfer size.**

Caveat on $0.236 actual versus $0.204 modelled: the table assumes 0.23 h of
rental plus download time, while this run billed 0.308 h. The four-minute apt
stall is in the real number and not in the model.

### What this changes

- For repeated work, keep the box and amortize: the second video costs 9 m 50 s
  and no provisioning.
- For genuinely one-shot work, pick the host by `search_cheap_egress.py`, not by
  `$/hr` or link speed, and use the s5-lineage workflow so you skip R2's warmup.
- Do not quote per-render times as if they were the cost of a video. On this
  rental the render was 60% of the clock and 63% of the money.

## Reminder on the ceiling

None of these reach the API provider's ~6–7 min. That's near-certainly multi-GPU Ulysses sequence parallelism (officially supported: `torchrun --ulysses_size=N --dit_fsdp`), which the single-GPU kijai wrapper can't do. To actually match the API you'd leave ComfyUI and run `generate_infinitetalk.py` under torchrun on a multi-GPU box. Separate experiment.
