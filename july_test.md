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

### Quality — assessed 2026-07-27

**4-step does not cost visible quality.** Measured with
`scripts/quality_metrics.py` over the full clips, plus frame strips from
`scripts/compare_quality.py` in `notes/quality/`:

| | edge energy (sharpness) | frame-to-frame delta |
|---|---|---|
| 6step dpm++_sde sdpa | 35.81 | 1.122 |
| 4step distill sdpa | 35.63 | 1.048 |
| 4step distill sage | 35.64 | 1.033 |

Sharpness differs by **0.5%** — noise. Frames at 1/3/5/7 s show identical
composition, lighting, beard and fur detail; an amplified difference map is flat
across background, tree and desk, with residual only on the subject's silhouette,
i.e. pose drift rather than texture loss.

**SSIM between 4-step sdpa and 4-step sage is 0.973.** Same seed, same sampler,
only the attention kernel differs, so this is the cleanest available read on
whether SageAttention changes output: it does, slightly — it is quantized
attention, not a bit-exact reimplementation — but not visibly.

Two caveats on these numbers:

- The frame-to-frame delta conflates flicker with how much the subject actually
  moves. 4-step scoring lower is **not** evidence of better temporal stability;
  a clip with less head motion scores the same way. Do not quote it as a win.
- SSIM between 6-step and 4-step is 0.896, and that figure means nothing about
  quality. Different scheduler and step count means the denoising trajectories
  diverge from the first step, so it measures divergence. It is recorded only to
  stop someone computing it later and reading it as degradation.

These comparisons are valid because every sweep workflow starts from the same
base seed and `bump_seeds()` is deterministic (`base + run*1000 + 7`), so run 2
of every variant sampled identical noise. Both variants also carry the same
distill LoRA at strength 1, so the LoRA is not a confound; what differs between
the 4-step and 6-step arms is step count **and** scheduler
(`flowmatch_distill` vs `dpm++_sde`), which cannot be separated — the distill
scheduler is what makes 4 steps viable.

**Verdict: adopt 4-step.** The 2.41x is free.

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

The pairs that isolate one variable each (this guidance was previously stated
backwards — `00014` vs `00012` differs in step count *and* attention backend, so
it answers neither question cleanly):

- **Step count + scheduler:** `00006` (4-step distill, sdpa) vs `00012` (6-step
  dpm++_sde, sdpa) — attention held constant.
- **Attention backend:** `00006` (4-step sdpa) vs `00014` (4-step sage) —
  sampler held constant.

Both were assessed above; 4-step passes.

### Measurement trap worth keeping

Re-queueing a byte-identical workflow does not regenerate anything: ComfyUI
caches on node inputs and returns the previous result. This showed up as a
"3.0 s generation" that emitted the same filename as the 171.5 s run before it.
`run_july_tests.py` now bumps every `seed`/`noise_seed` per run, which forces
real resampling while leaving the torch.compile cache warm — which is exactly
what run 2 is meant to measure.

## Decision

**Resolved 2026-07-27: 4step passed quality — it is the default, the 6-step
baseline is retired.** Sharpness within 0.5% of 6-step, no visible artifacts,
and SageAttention is visually equivalent to sdpa at SSIM 0.973.

The contingencies that no longer apply, kept because they are the right ladder
if a future model or resolution regresses:

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

### The cold-compile penalty is at least ~75 s, not ~33 s

The render took 664.8 s against 589.9 s for the same workflow on a warm inductor
cache. The ~33 s cold/warm delta measured on an 8 s clip **does not carry over** —
a longer clip exercises more graph variants, so more gets compiled. Do not
extrapolate warmup cost from short clips.

**Treat 75 s as a floor.** These two runs differ in two variables, not one. The
589.9 s figure was measured *before* the WANOPT patch was ever enabled on that
rental — it emitted clip `00016`, and `finish_480p.log` shows the flags being set
for the first time only after s8 finished (clip `00019`), the patch being inert
until then because it is env-gated. The e2e run had both `WANOPT_Y_CACHE` and
`WANOPT_KEEP_CACHE_WARM` set automatically at provisioning. So R3/R6 was making
the *cold* run faster while the warm run had no such help, which compresses the
measured gap. Adjusting by the ~9.4 s those flags are worth on an 8 s clip
suggests a true cold penalty nearer **~140 s**.

I concluded from this that `torch.compile` still pays on a single 58 s video,
"saving roughly 10 s per window over 21 windows (~215 s) against a ~140 s cold
cost". **That was extrapolated from an 8 s clip and it is wrong.** The same-box
A/B has now been run — see below — and `max-autotune` is a net *loss* on one
video.

The warmup that genuinely does *not* pay back one-shot remains **R2**, the VAE
decoder compile, at 646.5 s — which is why `full58_sage_API.json` (the s5
lineage) is the right workflow for this mode and s8 is not.

Provisioning was 5 m 59 s, of which **4 m 01 s was `apt install`** — host
variance, not a regression. SageAttention cost 12 s: the cached `sm_120` wheel
passed its probe and the source build was cancelled. The 34 GB of models pulled
in 1 m 38 s on a 7,944 Mb/s link. Both WANOPT flags were set automatically, so
this run had the s9 configuration.

## torch.compile A/B — 2026-07-27, one box, vast.ai 45971600

Every workflow here wires `compile_args`, and the sweep variants named "nocomp"
only switch `mode` to `default`, so compiling had never been tested against not
compiling. Three arms on one Nebraska 5090, each a full 58 s clip, ComfyUI
restarted and `/tmp/torchinductor_root` wiped between arms:

| arm | `mode` | render | cache before → after |
|---|---|---|---|
| a_autotune | `max-autotune-no-cudagraphs` (ships) | **591.8 s** | 0 → 360 MiB |
| b_default | `default` | 492.7 s ⚠️ | 360 → 370 MiB |
| c_nocompile | `compile_args` removed | **579.5 s** | 370 → 370 MiB |

**A vs C is the sound comparison and it says autotune loses by 12.3 s.** Arm A
started from an empty cache; arm C compiles nothing at all. Autotuning benchmarks
kernel variants on the GPU at first call, and across 21 windows the kernels it
picks never repay the search. It pays from the second render on — which a
one-shot rental never reaches.

**B is contaminated and I caused it.** The first version of
`scripts/e2e_compile_ab.sh` did not wipe the inductor cache between arms, on an
assumption I never verified: that inductor keys entries by compile mode, so the
arms could not share them. It does not, or not sufficiently — arm B started with
arm A's 360 MiB already on disk and grew it by only 10 MiB. "Cold default" and
"warm autotune" are indistinguishable in that number. The script now wipes the
cache; a single-arm rerun was launched and cancelled before it produced a result.
**Treat 492.7 s as an upper bound on how good `default` can look.**

Actionable today: **drop `max-autotune-no-cudagraphs` for one-shot rentals.**
Whether `default` beats no-compile is unresolved and needs one clean run.

Harness: `scripts/e2e_compile_ab.sh`, arms generated by
`scripts/build_compile_ab.py` from `full58_sage_API.json` (node 177
`WanVideoTorchCompileSettings`, node 122 `WanVideoModelLoader`). Cost $0.331.

## Output smoothness — 2026-07-27

The 58 s output did not feel smooth. Four hypotheses, one survived.

**Dead: duplicate frames.** Exact `framemd5` gives **1521/1521 unique frames**.
An earlier `mpdecimate` pass claimed 191 duplicates and an effective 21.9 fps —
a false positive I published and retracted the same day. `mpdecimate` is built
for cuts and letterboxing and collapses frames where only a mouth moves against a
static background, which is this workload exactly. **Do not use it on talking
heads.** Also pass `-map 0:v` to `framemd5` or it hashes the audio stream and
reports 4141 "frames".

**Dead: window-seam judder.** Motion measured at strides 64/68/72/76/81 varies by
at most 1.09x. The `motion_frame=9` overlap blends cleanly.

**Dead: motion decay.** Per-decile means 0.88 / 0.81 / 0.78 / 0.73 / 0.79 / 0.71
/ 0.69 / 0.86 / 0.76 / 0.55 — noisy, not a trend.

**Dead: 4-step being choppier than 6-step.** Jerk-over-motion is **0.180 for
4-step against 0.199 for 6-step**. The distilled model is smoother, not rougher.

**Real: a 2.64 s dead tail.** The video is **60.84 s against 58.20 s of audio** —
1521 frames where the speech needs 1455. Motion in the tail averages 0.30 against
0.78 for the body. It reads as the video hanging after the voice stops.

`VHS_VideoCombine.trim_to_audio` was `false`, so the node kept every frame the
sampler produced. InfiniteTalk works in 81-frame windows and always rounds *up*
past the end of the audio, so the overhang is structural rather than a one-off.

**Fixed:** `trim_to_audio: true` in `IT_{4090,5090}_july2026_{4,5}step.json` and
`workflows/generated/full/full58_sage_API.json`, applied by
`scripts/set_trim_to_audio.py`. The archived sweep and `compile_ab` workflows are
left as they are — they record runs that happened. **Verified in the JSON only;
not yet re-rendered.**

Scripts: `scripts/check_smoothness.py`, `scripts/quality_metrics.py`.

### Host selection for one-shot is not what the search optimizes

This run billed **$0.262** (`vastai show invoices-v1 --charges`): gpu 0.3452 h x
$0.480 = $0.166, disk 0.3491 h x $0.017 = $0.006, bwd 34.4 GB x $0.0026 = $0.090.
Egress was **34% of the bill**, the GPU 63%.

I had modelled $0.236 — 11% low. Two lessons: **vast billed 0.345 h against
0.308 h of wall clock** (billing starts before ssh answers and does not stop the
instant destroy is issued, so budget ~10% over the stopwatch), and **disk is its
own line item** which I had omitted. Note `--start-date` alone throws a
`TypeError` in the CLI; always pass both dates.

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
egress saves ~$0.078. That looked like "cheap egress beats a fast link at this
transfer size" — and then it cost 14 minutes.

### …and ranking on money alone bought a 14-minute stall

That table ranks on $. It picked machine `69187` (Hong Kong, $0.401/hr,
$1.30/TB, advertised 1,311 Mb/s) at a modelled $0.160. The instance **sat in
`loading` for 14 m 11 s, never reached ssh, produced nothing, and was destroyed
while still pulling the container image.**

Three defects, all now fixed in `search_cheap_egress.py` *and*
`interactive_search_vastai.py`:

- **The container image was never in the model.**
  `vastai/comfy:v0.28.0-cuda-12.9-py312` is **9.53 GB** compressed and is pulled
  on every fresh rental, before provisioning and before ssh answers. The billed
  `bwd` was 34.4 GB — the models alone — so **vast does not charge egress on the
  image pull**. It costs time but not money, which is exactly why a $-ranked
  search could not see a 14-minute stall. Real pull is 43.5 GB, not 34.
- **Advertised `inet_down` predicts little.** 1,311 Mb/s puts 9.5 GB under a
  minute; fourteen minutes was not enough. Machine `54134` at 1,699 Mb/s had ssh
  up in 52 s. Both scripts now derate advertised speed to **35%** and enforce a
  **2,500 Mb/s floor** — to exclude hosts that cannot serve a fast pull, not to
  chase the fastest link.
- **Neither ranking priced waiting.** Both now add `minutes x $0.02` to the
  estimate (`TIME_VALUE_USD_PER_MIN`, `--time-value`), so five cents buys about
  two and a half minutes. Hard ceilings (`--min-speed`, `--max-cost-per-tb`,
  `--max-cost`) sit above it so speed cannot be bought at any price. Set
  `--time-value 0` to get the old money-only ranking back.

`MIN_INET_DOWN_SPEED` went **5000 → 2500** in the same change: at 5,000 every
surviving host charged $2.60–$10.00/TB, which is what started this.

What the reworked ranking picks — fast *and* cheap:

| machine | $/hr | $/TB | down | pull | one-shot | score |
|---|---|---|---|---|---|---|
| 99580 (UK) | 0.606 | **0.98** | 4,225 Mb/s | 3.9 min | $0.212 | **0.567** |
| 141234 (CN) | 0.535 | 2.60 | 4,324 Mb/s | 3.8 min | $0.246 | 0.598 |
| 45707 (Iceland) | 0.668 | 2.73 | 4,639 Mb/s | 3.6 min | $0.286 | 0.634 |

### What this changes

- For repeated work, keep the box and amortize: the second video costs 9 m 50 s
  and no provisioning.
- For genuinely one-shot work, pick the host by `search_cheap_egress.py`, and use
  the s5-lineage workflow so you skip R2's warmup.
- **Drop `max-autotune` for one-shot.** It is measurably slower than not
  compiling at all on a single 58 s clip.
- Do not quote per-render times as if they were the cost of a video. On this
  rental the render was 60% of the clock and 63% of the money.
- Rank hosts on money *and* time. Either one alone picks badly, in opposite
  directions.

## Reminder on the ceiling

None of these reach the API provider's ~6–7 min. That's near-certainly multi-GPU Ulysses sequence parallelism (officially supported: `torchrun --ulysses_size=N --dit_fsdp`), which the single-GPU kijai wrapper can't do. To actually match the API you'd leave ComfyUI and run `generate_infinitetalk.py` under torchrun on a multi-GPU box. Separate experiment.
