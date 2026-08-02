# InfiniteTalk Docker Testing Environment

This project provides a local testing environment for InfiniteTalk Docker setup before deployment to vast.ai infrastructure.

## Project Overview

InfiniteTalk is an AI-powered video generation system that combines:
- Wan2.1-I2V-14B-480P video generation model
- Chinese Wav2Vec2 audio processing
- InfiniteTalk speech-to-video synchronization

This repository contains testing and setup scripts to verify the Docker environment locally before deploying to vast.ai.

## Directory Structure

- `SIMPLE_START.md` - Step-by-step Docker setup guide for testing
- `docker_data/` - Persistent storage for Docker container data
  - Stores downloaded model weights to avoid re-downloading
  - Contains configuration files and test outputs
  - Survives container restarts and recreation
- `InfiniteTalk/` - InfiniteTalk source code and models (gitignored)
- `scripts/` - Utility scripts for vast.ai operations
- `input_files/` - Input files for testing
- `.gitignore` - Git ignore rules
- `README.md` - This file

## Git Setup

This repository uses Git for version control. The InfiniteTalk directory is gitignored to avoid tracking large model files and source code that may be managed separately.

### Ignored Files/Directories
- `InfiniteTalk/` - Complete InfiniteTalk source code and models
- `.hf_home/` - HuggingFace cache directory
- `.venv-backups/` - Virtual environment backups
- `docker_data/` - Docker persistent data (may contain large files)

## Docker Setup

Follow the steps in `SIMPLE_START.md` to:
1. Download and start the vastai/pytorch Docker container
2. Verify pre-installed dependencies (PyTorch, CUDA, etc.)
3. Install additional requirements (xformers, flash-attn, etc.)
4. Download model weights to `docker_data/` directory
5. Test the complete InfiniteTalk pipeline

## Usage

```bash
# Start Docker container (mounts current directory to /workspace)
docker run -it --gpus all \
  -v $(pwd):/workspace \
  vastai/pytorch:2.4.1-cuda-12.4.1-py310-22.04 \
  /bin/bash

# Inside container, follow SIMPLE_START.md steps
cd /workspace
# ... run verification commands
```

## Requirements

- Docker with NVIDIA GPU support
- NVIDIA drivers with CUDA 12.4+ compatibility
- At least 30GB free disk space for model weights (see download sizes below)
- Fast internet connection for model downloads

## Provisioning Scripts

### povision_fp8.sh

Automated provisioning script for ComfyUI with Wan2.1 models in FP8 format.

**Total Download Size: ~29.94 GB**

#### Model File Breakdown:
- `Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors`: 15 GB (main diffusion model)
- `umt5-xxl-enc-bf16.safetensors`: 10 GB (text encoder)
- `Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors`: 2.5 GB
- `clip_vision_h.safetensors`: 1.1 GB
- `lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors`: 703 MB
- `MelBandRoformer_fp16.safetensors`: 435 MB
- `Wan2_1_VAE_bf16.safetensors`: 242 MB

#### ComfyUI Custom Nodes:
- ComfyUI-WanVideoWrapper
- ComfyUI-VideoHelperSuite
- ComfyUI-MelBandRoFormer
- ComfyUI-KJNodes

**Check current sizes:** Run `scripts/check_download_sizes.sh` to verify latest file sizes without downloading entire files.

## Next Steps

After successful local testing:
1. Deploy to vast.ai infrastructure
2. Configure production environment
3. Set up monitoring and logging
4. Optimize for performance

## Quick Start on vast.ai

```bash
# Install aria2 for faster downloads
apt-get update && apt-get install -y aria2

# Download and run provisioning script
wget https://raw.githubusercontent.com/daromaj/vast_experiments/refs/heads/master/povision_fp8.sh
chmod +x povision_fp8.sh
./povision_fp8.sh

# Copy outputs from vast.ai instance
vastai copy INSTANCE_ID:/workspace/ComfyUI/output local:output
```

## execution time for ~60s video

### 2026-07-26 — current numbers (RTX 5090, measured)

**58 s audio (1455 frames, 21 windows) @ 480x832: 9 m 50 s**, down from 13 m 29 s.

The old figures below were bottlenecked by a SageAttention wheel built for the
wrong GPU. It was cached under `python/sage/torch2.10.0-cu128-sm_120/` but
compiled for Ada, so on a 5090 sage fell back silently — correct output, but
>=6 GB more VRAM than SDPA, which OOMed `WanVideoSampler` at 29.3 GiB. Rebuilt
with `TORCH_CUDA_ARCH_LIST=12.0` it is the fastest backend by a wide margin.
A wheel's filename tells you nothing about its target: SageAttention 2.2.0 always
emits `_qattn_sm80` / `_qattn_sm89` because those are the kernel *sources*.

8 s clip, run 2 of 2 (run 1 discarded — it pays the torch.compile warmup):

| Config | Attention | Steps | Gen time | vs baseline |
|---|---|---|---|---|
| sage + 4-step + quick wins + node patch | sageattn | 4 | **72.3 s** | **2.93x** |
| sage + 4-step + quick wins | sageattn | 4 | 81.7 s | 2.59x |
| sage + 4-step distill | sageattn | 4 | 87.7 s | 2.41x |
| sdpa + 4-step distill | sdpa | 4 | 117.8 s | 1.80x |
| old 6-step baseline | sdpa | 6 | 211.5 s | — |

The 9 m 50 s full-clip figure above predates the last two rounds; it was measured
at the 87.7 s config. The 72.3 s config should land nearer 8 minutes but that has
not been measured on a full 58 s clip — do not quote it as if it had been.

Isolated attention kernel at the real shape (40 heads x 32,760 tokens x 128 dim):
**sageattn 39.1 ms vs sdpa 106.9 ms**.

**Quality checked 2026-07-27 — the 2.93x is free.** 4-step distill scores within
**0.5%** of the 6-step baseline on edge energy (sharpness) across the full clip,
with no visible artifacts in matched frames and difference maps flat everywhere
except the subject's silhouette. SageAttention vs sdpa at identical sampler and
seed: **SSIM 0.973** — quantized attention does change the output slightly, just
not visibly. Details, caveats and the metrics that do *not* mean what they look
like: `july_test.md`. Frame strips in `notes/quality/`.

(`notes/` holds run artifacts — contact sheets, per-run JSON, VRAM traces. It is
gitignored and kept local, so paths under it referenced here and in `july_test.md`
exist in a working copy that has done the runs, not in a fresh clone.)

720p (1280x720) does fit on a 5090 at `blocks_to_swap=0` — 27.8 GiB peak with
`tiled_vae=true` — but costs ~4.4x wall-clock, and there is no 720p lightx2v
distill LoRA published (only 480p ranks exist), so the 4-step LoRA is a mismatch
at that resolution. See `july_test.md`.

Full method, per-setting attribution and the source audit behind the quick wins:
`july_test.md` and `notes/node_optimization_audit.md`.

### 2026-07-27 — RTX 4090 (measured)

**58 s audio @ 480x832 on a 4090: 13 m 36 s** (816.3 s, peak 20,626 MiB), against
**9 m 50 s** on a 5090. The 4090 is **1.38x slower** on the same clip.

The 4090 needs block swapping and the 5090 does not. Its 24,564 MiB nameplate is
not the budget — torch reports a **device limit of 23.52 GiB**, with ~480 MB going
to driver and context. The pipeline peaks at ~24.4 GiB on a 5090, so the 4090 is
~0.9 GiB short of running resident.

The interesting result is that **more block swapping is faster, not slower**:

| swap | warm (8 s clip) | peak reserved | |
|---|---|---|---|
| 2 | — | 24,068 MiB (pinned) | **OOM** |
| 4 | 254.9 s | 24,080 MiB (pinned) | |
| 6 | 255.8 s | 24,076 MiB (pinned) | |
| 12 | 108.9 s | 21,782 MiB | |
| **16** | **105.2 s** | 20,562 MiB | **fastest** |
| 20 | 106.9 s | 20,562 MiB | shipped default |
| 28 | 114.5 s | 20,562 MiB | transfers now dominate |

swap=4 and swap=6 both sit pinned at 24,07x MiB, which *is* the 23.52 GiB device
limit, and both run ~2.4x slower than swap=16. Pinned against the ceiling the
caching allocator thrashes — synchronous frees and `cudaMalloc` retries — and that
costs far more than moving another dozen blocks across PCIe. The curve is
U-shaped with a flat floor from 12 to 20 and only turns back up at 28.

**The minimum config that fits is the worst one to run.** Sizing block swap by
"how much must I shed to avoid OOM" produces exactly the 4-6 range, which is the
slowest working configuration on the card.

`blocks_to_swap=20` in `workflows/IT_4090_july2026_*.json` is therefore **correct
and unchanged** — it is 1.6% off the measured optimum of 16, which is noise, and
both report identical peak VRAM. It was a lucky default (it is the node's own),
but it is the right one.

Note the peak column is *reserved*, not required: PyTorch's caching allocator
keeps what it takes, so it distinguishes "had headroom" from "had none" and
nothing finer.

This also inverts the pre-fix 4090 figures further down (`swap 5 = 238 s` beating
`swap 20 = 314 s`). That ordering was an artifact of the broken SageAttention
wheel; with a correctly built one it reverses.

**Cost per 58 s video**, at the cheapest hosts meeting the search criteria:

| | RTX 5090 | RTX 4090 |
|---|---|---|
| cheapest $/hr seen | $0.456 | $0.362 |
| 58 s render | 589.9 s | 816.3 s |
| **$/video** | **~$0.075** | **~$0.082** |

The 4090's ~21% lower hourly rate does not cover its ~38% longer render, so the
5090 is both cheaper per video and faster in wall-clock. It also has headroom for
720p and never enters the thrash regime. **Prefer the 5090**; the 4090 is a
fallback on availability.

### Provisioning time (create to first queued job)

| Stage | 4090 | 5090 |
|---|---|---|
| container pull + boot | 1 m 26 s | not captured |
| apt + pip base | 0 m 41 s | 0 m 48 s |
| model downloads + node install | 1 m 49 s | 2 m 23 s |
| SageAttention | **2 m 04 s (source build)** | 0 m 10 s (cached wheel) |
| WANOPT node patch | 0 m 07 s | n/a |
| **script total** | **4 m 54 s** | 3 m 21 s |
| **create to queueable** | **~6 m 30 s** | ~4 m 45 s |

The whole gap is the sage build. The 5090 skipped it because a wheel was cached
for its ABI; a matching `sm_89` wheel is now committed under
`python/sage/torch2.10.0-cu128-sm_89/`, which should bring the 4090 to ~4 m 20 s.

Queueable is not the same as having a video. The first generation pays the
torch.compile warmup — run 1 versus run 2 above is 230.4 s versus 105.2 s on an
8 s clip. For a **one-shot rental** (provision, render once, destroy) that warmup
is pure overhead: measured on one box, `max-autotune` is *slower* than not
compiling at all on a single 58 s clip. See the A/B below.

### 2026-07-27 — one-shot rental, end to end (measured)

Rent a 5090, make one 58 s video, destroy it. **18 m 29 s of billed wall clock**,
instance `45967149` on a $0.481/hr host. Every number below is a stamp taken as
the run happened (`output/e2e_*/phases.tsv`), not scraped from logs afterwards.

| Phase | Duration |
|---|---|
| create → ssh reachable | 0 m 49 s |
| provisioning | 6 m 27 s |
| upload workflow + assets | 0 m 04 s |
| **render 58 s clip (cold cache)** | **11 m 06 s** |
| download outputs | 0 m 02 s |
| destroy | 0 m 01 s |
| **total** | **18 m 29 s** |

Output: 60.84 s, 480x832, 1521 frames, 8.6 MB. Peak VRAM 26,474 MiB — higher
than the ~24.4 GiB seen earlier because `WANOPT_KEEP_CACHE_WARM=1` skips the
per-window `soft_empty_cache()`, so the allocator keeps more. Still comfortable
on 32 GB.

**Only 60% of the bill is generation.** The other 7 m 20 s is provisioning and
boot, paid once per video in this mode instead of amortised over a sweep.

**The cold compile costs *at least* ~75 s, not ~32 s.** The render was 664.8 s
against 589.9 s for the same workflow on a warm inductor cache, so the 32.4 s
cold/warm delta measured on an 8 s clip does not carry over — a longer clip
compiles more graph variants. That 75 s is a floor: the 589.9 s run predates the
WANOPT patch being enabled, so R3/R6 was speeding up the *cold* run and dragging
the apparent gap down.

What that comparison could not answer is whether compiling is worth anything at
all on a one-shot rental. It is not — see the same-box A/B below, which
supersedes the "compile saves ~215 s across 21 windows" figure previously
claimed here. That number was never measured; it was extrapolated from an 8 s
clip and is wrong.

The other warmup that does not pay back on a single video is s8/R2, the VAE
decoder compile, at 646.5 s.

Provisioning was 5 m 59 s of which **4 m 01 s was `apt install`** — host
variance, not a regression. SageAttention cost 12 s: the cached `sm_120` wheel
passed its probe and the source build was cancelled. Model download was 1 m 38 s
for 34 GB on a 7,944 Mb/s link.

#### Cost — billed, not modelled

`vastai show invoices-v1 --charges` returns what was actually taken. For this run
(`45967149`), **$0.262**:

| line | quantity | rate | amount |
|---|---|---|---|
| gpu | 0.3452 h | $0.480/h | $0.166 |
| disk | 0.3491 h | $0.017/h | $0.006 |
| bwd (download in) | 34.4 GB | $0.0026/GB | $0.090 |
| bwu (upload out) | ~0 | | $0.000 |
| **total** | | | **$0.262** |

I had modelled $0.236 — **11% low**, from two mistakes worth remembering:

- **vast bills 0.345 h against 18 m 29 s (0.308 h) of wall clock.** Billing
  starts before ssh answers and does not stop the instant destroy is issued;
  budget ~10% more hours than the stopwatch shows.
- **disk is its own line**, charged for slightly longer than the GPU, and I had
  omitted it entirely.

The itemisation also confirms what dominates: **egress is 34% of the bill**, and
the GPU only 63%.

Note the CLI's date handling — `--start-date` alone throws a `TypeError` (it does
`args.start_date + 7*24*60*60` on a string). Always pass both dates.

#### Why the search criteria were wrong

`interactive_search_vastai.py` filters on `inet_down >= 5000` Mb/s, which for a
one-shot rental is backwards: the 34 GB of models is paid *per video*, so $/GB
matters more than link speed, and every host passing that filter charged
$2.60–$10.00/TB. Ranking the same GPUs by full one-shot cost instead
(`scripts/search_cheap_egress.py`):

| machine | $/hr | $/TB | down Mb/s | one-shot |
|---|---|---|---|---|
| 32637 (Alberta) | 0.482 | **0.33** | 1,834 | **$0.142** |
| 54134 (Nebraska) | 0.534 | **0.00** | 1,699 | $0.147 |
| 108568 (Malaysia) | 0.401 | 1.30 | 1,377 | $0.159 |
| 141151 (the host used here) | 0.481 | 2.60 | 7,944 | $0.204 |

A 4x slower link adds ~1 min of download, worth ~$0.015 of rental, while the
cheaper egress saves ~$0.078. **Cheap egress beats a fast link**, and the
ordering survives charging each host for its own download time.

#### …and then cheap egress overcorrected, and cost 14 minutes

That model ranks on money. It picked machine `69187` (Hong Kong, $0.401/hr,
$1.30/TB, advertised 1,311 Mb/s) at a modelled $0.160. The instance **sat in
`loading` for 14 m 11 s, never reached ssh, and produced nothing.** It was still
pulling the container image when it was destroyed.

Three things were wrong, all now fixed in both search scripts:

- **The container image was not in the model at all.**
  `vastai/comfy:v0.28.0-cuda-12.9-py312` is **9.53 GB** compressed, pulled on
  every fresh rental, *before* provisioning starts and before ssh answers. The
  billed `bwd` on the e2e run was 34.4 GB — the models alone — so vast **does not
  charge egress on the image pull**. It costs time but not money, which is
  precisely why a $-ranked search was blind to it. Total pull is 43.5 GB, not 34.
- **Advertised `inet_down` is a weak predictor.** 1,311 Mb/s puts 9.5 GB at under
  a minute; it had not finished in fourteen. Machine `54134` at 1,699 Mb/s had
  ssh up in 52 s. Both scripts derate advertised speed and enforce a floor — the
  floor exists to drop hosts that cannot serve a fast pull, not to chase the
  fastest link on the market. (The derate was 35% and the floor 2,500 Mb/s when
  this was written; both were recalibrated on 2026-08-02 to 90% and 1,500 Mb/s
  after the measurement behind them turned out to be wrong — see below.)
- **Neither ranking priced the wait.** Both now add
  `minutes x $0.02` (`TIME_VALUE_USD_PER_MIN` / `--time-value`) to the estimate,
  so five cents buys about two and a half minutes. Ranking on $ alone picks hosts
  that stall; ranking on time alone invites paying for a marginally faster link.
  Hard ceilings (`--min-speed`, `--max-cost-per-tb`, `--max-cost`) sit on top so
  speed cannot be bought at any price.

`MIN_INET_DOWN_SPEED` in `interactive_search_vastai.py` went **5000 → 2500** in
the same change: at 5,000 every surviving host charged $2.60–$10.00/TB, which is
what started this whole detour.

What the reworked ranking picks now — fast *and* cheap, rather than either:

| machine | $/hr | $/TB | down | pull | one-shot | score |
|---|---|---|---|---|---|---|
| 99580 (UK) | 0.606 | **0.98** | 4,225 Mb/s | 3.9 min | $0.212 | **0.567** |
| 141234 (CN) | 0.535 | 2.60 | 4,324 Mb/s | 3.8 min | $0.246 | 0.598 |
| 45707 (Iceland) | 0.668 | 2.73 | 4,639 Mb/s | 3.6 min | $0.286 | 0.634 |

### 2026-07-27 — torch.compile does not pay for itself on one video (measured)

Every workflow in the repo wires `compile_args`, and the two variants named
"nocomp" in the sweeps only switch `mode` to `default` — so nothing here had ever
tested compiling against *not* compiling. Three arms, **one box** (`45971600`,
Nebraska 5090), each rendering the full 58 s clip, `/tmp/torchinductor_root`
wiped between arms:

| arm | `mode` | render | inductor cache before → after |
|---|---|---|---|
| a_autotune | `max-autotune-no-cudagraphs` (ships today) | 591.8 s | 0 → 360 MiB |
| b_default | `default` | 492.7 s ⚠️ | 360 → 370 MiB |
| c_nocompile | `compile_args` removed | 579.5 s | 370 → 370 MiB |

**`max-autotune-no-cudagraphs` is 12.3 s slower than not compiling at all.** That
comparison is sound: arm A started from an empty cache and arm C compiles
nothing, so both are honest cold numbers. Autotuning benchmarks kernel variants
on the GPU at first call, and across 21 windows the winning kernels never earn
back what the search cost. It pays from the *second* render onward, which a
one-shot rental never reaches.

**The 492.7 s for `mode=default` is not trustworthy** — my error, not the
harness's. The first version of the script did not wipe the inductor cache
between arms, on an unverified assumption that inductor keys entries by mode. It
does not, or not enough: arm B started with arm A's 360 MiB already on disk, so
"cold default" cannot be told apart from "warm autotune". The script now wipes
the cache (`rm -rf /tmp/torchinductor_root`) and a single-arm rerun was launched
to settle it, but was cancelled before it produced a number. **Treat 492.7 s as
an upper bound on how good `default` can look, not as its cold cost.**

What is safe to conclude today: **stop shipping `max-autotune` for one-shot
rentals.** Whether `default` beats no-compile is open.

The three arms cost $0.331 billed.

### 2026-08-02 — settled: `mode=default` is 27% faster, and compile is worth keeping

Rerun of the above with the cache actually wiped before every arm, on one Poland
5090 (`46600649`, machine 144163, $0.481/hr), full 58 s clip each:

| arm | `mode` | render | inductor cache after | vs shipped |
|---|---|---|---|---|
| a_autotune | `max-autotune-no-cudagraphs` (shipped) | 659.2 s | 359 MiB | — |
| **b_default** | **`default`** | **480.4 s** | **18 MiB** | **−178.8 s (−27.1%)** |
| c_nocompile | `compile_args` removed | 564.4 s | 0 MiB | −94.8 s (−14.4%) |

Both July conclusions were right in direction and wrong in size. Autotune loses
by **178.8 s, not 12.3 s** — the July figure came from A-vs-C, which measures
autotune against no-compile and skips the option that actually wins. And
compile *is* worth keeping: `default` beats no-compile by 84.0 s. The 359 MiB
versus 18 MiB cache shows why — autotuning spends three minutes benchmarking
kernel variants across 21 windows, and only earns it back on a second render
that a one-shot rental never performs.

No quality cost. Edge energy 35.2491 / 35.2520 / 35.2512 — a 0.008% spread.
Jerk-over-motion 0.2341 / 0.2334 / 0.2482, so `default` matches autotune and
only no-compile is measurably rougher. All three: 1454 frames, 58.200998 s.
Pairwise SSIM 0.9793 / 0.9743 / 0.9733 is mutually equidistant — fp jitter
reordering kernels, not degradation. (January-vs-current was 0.896, and that
one was visible.)

**Applied:** `workflows/generated/full/full58_sage_API.json` now ships
`mode: default`.

$0.38 for all three arms, 34.5 min create→destroyed.

### 2026-08-02 — the shipped config, rendered end to end

One video with everything applied — `mode: default`, deferred CUDA toolchain,
the fixed host ranking — on machine `104900` (Massachusetts, $0.499/hr,
**$0.00/TB egress**, advertised 9,135 Mb/s), instance `46608622`:

| phase | duration |
|---|---|
| create → ssh reachable | 1m 36s |
| provisioning (models, nodes, sage) | 5m 36s |
| upload workflow + assets | 0m 16s |
| render 58 s clip, cold cache | 8m 31s |
| download outputs | 0m 09s |
| destroy | 0m 01s |
| **total billed wall clock** | **16m 09s** |

Output: 1454 frames, 58.200998 s, 480×832, peak VRAM 25,224 MiB — frame count
and duration identical to all three A/B arms. Render 507.7 s against the A/B's
480.4 s for the same config, +5.7% on a different box.

#### Four hosts in a row could not pull the container image

Hong Kong, California and Virginia ×2 all failed, two with
`failed to resolve reference ... not found` and one with
`Get "https://registry-1.docker.io/v2/": net/http: request canceled`. **The tag
was fine** — a fresh token against `registry-1.docker.io` returned HTTP 200 for
`v0.28.0`, `v0.29.0` and `v0.29.2`. So bumping the image would not have helped
and would have thrown away the cached SageAttention wheel. The 200s came from a
residential IP and prove nothing about what a datacenter host can reach; the
remaining explanation is a pull path shared by vast hosts, not a stale pin.

Each bad host was abandoned in **~90 s** instead of waiting out the 20-minute
ssh timeout, at a cost of about $0.009 apiece.

#### The fault detector had never worked

`instance_fault()` was a `python3 -c '...'` heredoc, and the Python inside it
contained single-quoted strings (`'unknown'`). Those quotes closed the *shell's*
quoted argument, so the interpreter received truncated source and answered
``SyntaxError: '{' was never closed`` on every poll. Machine `140178` burned a
22-minute rental doing exactly this, ~$0.12, and still fell through to the
timeout the detector exists to prevent.

`test_instance_fault.py` reported PASS throughout. It regex-extracted the Python
and ran it through `subprocess` with no shell involved — grading source text that
production never executed. That is worse than having no test: it converts
"untested" into "verified".

The classifier now lives in `scripts/instance_fault.py`, where there is no
quoting surface, and the test runs **bash** over the real function text with
`vastai` stubbed on `PATH`. Pointed at the previous revision via `E2E_SCRIPT` it
fails 17/17 with the production `SyntaxError`.

### 2026-08-02 — correction: the bandwidth calibration measured the wrong interval

Achieved throughput was computed from the *absolute* offset of the `[PHASE]
downloads finished` line, as though downloads began at t=0. They begin when apt
finishes. Under the old provisioning script the blocking 2 GB CUDA install ran
first, so the 7,944 Mb/s run was charged 3m32s of apt against its model pull —
340 s instead of the real 98 s — and reported 799 Mb/s for a host that actually
delivered 2,774. Every constant in the cost model was fitted to those numbers.

Corrected, measuring `downloads starting` → `downloads finished`, n=6
(regenerate with `scripts/calibrate_bandwidth.py`):

| advertised | window | achieved | % of advertised |
|---|---|---|---|
| 1,699 Mb/s | 166 s | 1,638 Mb/s | 96.4% |
| 7,318 Mb/s | 86 s | **3,161 Mb/s** | 43.2% |
| 7,398 Mb/s | 205 s | 1,408 Mb/s | 19.0% |
| 7,944 Mb/s | 98 s | 2,774 Mb/s | 34.9% |
| 8,021 Mb/s | 159 s | 1,710 Mb/s | 21.3% |
| 9,135 Mb/s | 261 s | **1,106 Mb/s** | 12.1% |

Among hosts advertising ≥3,000 Mb/s, achieved throughput spans 1,106–3,161 Mb/s
— a 2.9× spread — and **does not track the advertised figure at all**. The
9,135 Mb/s host was the slowest of the six; the 7,318 Mb/s host was the fastest.
The old claim "a 1,699 Mb/s host beat a 7,944 Mb/s one" is false on the
corrected numbers: 1,638 versus 2,774.

The policy conclusion survives and is better supported than before — above the
floor, advertised `inet_down` is noise, so **rank on price**. `test_host_ranking`
still passes unchanged, because price dominates the score.

`PIPELINE_CEILING_MBPS` (1300) became `OBSERVED_MEDIAN_SHARE_MBPS` (1700). The
old name asserted a mechanism that does not exist: it assumed a limit in the
download pipeline, when the cause is contention on the machine's shared uplink.
With a 2.9× spread it predicts no individual host well; it is kept only because
ranking needs a number in the formula.

### 2026-07-27 — output smoothness, and the one real defect

The 58 s output felt like it was not playing smoothly. Four hypotheses, three
dead:

- **Duplicate frames?** No. Exact `framemd5` hashing gives **1521/1521 unique
  frames, zero duplicates** — the container's 25 fps is the real rate. An earlier
  `mpdecimate` pass claimed 191 duplicates and 21.9 effective fps; that was a
  false positive. `mpdecimate` is tuned for cuts and letterboxing and collapses
  frames where only a mouth moves against a still background, which is exactly
  what this workload produces. Do not use it on talking heads. (And pass
  `-map 0:v` to `framemd5`, or it hashes the audio stream too and reports 4141
  "frames".)
- **Judder at window seams?** No. Per-frame motion measured at strides 64/68/72/
  76/81 shows a max ratio of 1.09 — the `motion_frame=9` overlap blends cleanly.
- **Motion decaying over the clip?** No clear trend. Per-decile means run
  0.88 / 0.81 / 0.78 / 0.73 / 0.79 / 0.71 / 0.69 / 0.86 / 0.76 / 0.55 — noisy,
  not monotonic.
- **4-step being choppier than 6-step?** The opposite. Jerk-over-motion is
  **0.180 for 4-step against 0.199 for 6-step** — the distilled model is
  *smoother*.

**The defect is a dead tail.** The video runs **60.84 s against 58.20 s of
audio** — 1521 frames generated where the speech needs 1455. The extra **2.64 s**
is the character idling after the voice stops, and motion there averages 0.30
against 0.78 for the body of the clip. It reads as the video hanging.

Cause: `VHS_VideoCombine.trim_to_audio` was `false`, so the node kept every frame
the sampler produced instead of cutting to the audio. InfiniteTalk works in
81-frame windows, so it always rounds *up* past the end of the audio — the
overhang is structural, not a one-off.

**Fixed:** `trim_to_audio: true` in the shipped workflows
(`IT_{4090,5090}_july2026_{4,5}step.json` and
`workflows/generated/full/full58_sage_API.json`), via
`scripts/set_trim_to_audio.py`. The archived sweep and `compile_ab` workflows are
deliberately left alone — they document runs that actually happened, and
rewriting them would misrepresent those runs. This has not been re-rendered yet,
so the fix is verified in the JSON, not in an output file.

Measurement scripts: `scripts/check_smoothness.py`, `scripts/quality_metrics.py`.

### 2026-08-02 — January 2026 vs the current stack, measured head to head

Everything above compares *settings*. This compares what the January stack
actually produced against what ships now: two 5090 rentals in parallel, one full
58 s clip each, same image, audio, seed and prompts.

Reproducing January needed a **node pin**, not just old settings — the January
workflow will not load against the current wrapper, because node 137's class
`DownloadAndLoadWav2VecModel` was later deleted in favour of
`Wav2VecModelLoader`. Pinned to `339e0fe` (2026-01-23) it **imported cleanly
against ComfyUI v0.28.0** with no dependency work.

| | A january (6-step `dpm++_sde`) | B current (4-step distill) |
|---|---|---|
| render, 58 s clip, cold | 921.9 s | **591.7 s** |
| edge energy (sharpness) | 35.83 | 35.22 (-1.7%) |
| unique frames (`framemd5`) | 1454/1454 | 1454/1454 |

**1.56x faster** — different hosts, so not a clean speed figure, but their GPUs
are within 2% on dlperf and neither swaps blocks.

Also the first render carrying the `trim_to_audio` fix, which was previously
verified in JSON only: both clips run **58.200998 s against 58.20 s of audio**.
The 2.64 s dead tail is gone, confirmed in an output rather than a config.

**The interesting result is motion, and the whole-frame number lies about it.**
The current clip shows 25% less frame-to-frame change, which read alone suggests
a deader render. Split by region it does not:

| | january | current | |
|---|---|---|---|
| wall / tree / desk | 0.11-0.24 | 0.12-0.29 | noise floor, both |
| **chair posts, mug** | **0.61-1.10** | **0.34-0.72** | static props, **1.4-1.8x more in January** |
| face, hands | 0.92-5.69 | 0.47-4.47 | subject |

Neither arm drifts its background. But the chair posts and mug are static props
sitting clear of the subject, and January moves them 1.4-1.8x more — the chair
visibly wobbles. That portion of its higher motion score is a **defect, not
liveliness**, which means the whole-frame motion mean and the jerk/motion ratio
built on it cannot compare these two arms at all. The rest of the gap is smaller
hand movement, a reasonable trade on a talking head.

Lip-sync judged good in both by eye; there is still no metric for it here.

**Verdict: the current stack is the better render, not merely the faster one.**
Method, the full region table, the confound that remains unresolved (the two
arms feed different wav2vec weights), and two harness traps this run exposed:
`july_test.md`.

### Disk sizing

Measured on a live 5090 rental after a full provision + several renders:

| Item | Size |
|---|---|
| `/workspace/ComfyUI/models` | 32 GB |
| — `diffusion_models` (Wan 14B fp8 + InfiniteTalk) | 19 GB |
| — `text_encoders` (umt5-xxl) | 11 GB |
| — clip_vision / loras / vae / wav2vec2 / transformers | ~2.7 GB |
| `/venv` | 8.8 GB |
| torch inductor cache (grows with compiles) | 1.6 GB |
| custom_nodes + SageAttention build | 0.6 GB |
| **fixed setup total** | **~43 GB** |

Per generated video the marginal cost is tiny — ComfyUI writes both a silent
`.mp4` and an `-audio.mp4`:

| Videos @ 1 min | Output size | Total disk |
|---|---|---|
| 1 | ~16 MB | ~43 GB |
| 10 | ~160 MB | ~43 GB |

**So `--disk 60` is comfortable and `--disk 50` is workable; the models dominate
and video count is irrelevant at this scale.** Adding the 720P checkpoint would
push the fixed cost to ~60 GB, so budget `--disk 80` if you provision both.

### older figures (pre-fix, kept for reference)

Mon Dec  8 22:20:07 UTC 2025
** ComfyUI startup time: 2025-12-08 22:22:31.139

Prompt executed in 00:16:23

overall potentially under 20 minutes e2e for 60s video (on vastai instance with fast internet)

for instance with $0.60/hr this should be less than $0.30 per video

13:29 for 58s audio with sageattention 2 and 3 installed but regular sage attention selected

8 minutes just to get comfyui up and running

4090 - 314s for 10s video with sageattention 2 and block swap 20 ~ 32 minutes for 60s video
4090 - 238s for 10s video with sageattention 2 and block swap 5 ~ 24minutes for 60s video

58s Prompt executed in 00:49:55

we need to choose pcie4 for 4090
so far I was not able to fit models in 4090 memory

Also - if the host is not ready within a minute it's probably better to cancel the instance and try different one

4090
# Run this inside a standard PyTorch 2.5/CUDA 12.4 container
export TORCH_CUDA_ARCH_LIST="8.9"
python setup.py bdist_wheel

5090
# Run this on your CUDA 12.9 machine
export TORCH_CUDA_ARCH_LIST="12.0"

# Optional: Add a local version tag so you don't mix them up
export SAG_VERSION_SUFFIX="+cu129" 

python setup.py bdist_wheel

export TORCH_CUDA_ARCH_LIST="12.0"
export SAG_VERSION_SUFFIX="+cu129"

# Instead of "python setup.py bdist_wheel", use:
pip wheel . --no-deps -w dist/



---
# audio 5s

## 5090 PCIE 4.0/16x
sage attention 2 (~6-7s per it)
<102s
flash attention (~8-9s per it)
<116s

# audio 58s - 21 windows

## 5090

sage attention 2 (~6-7s per it) 34s per window
~12m
flash attention (~8-9s per it) 49s per window + ~5s rest
18:20

## 4090 PCIE 4.0/16x
sage attention 2 (~7-12s per it) 46s per window
18:45

flash attention (~11-15s per it) 1:09 per window + ~15s rest
26:14
~24:30

## 5090
[FLASH] Flash Attention installation complete. Duration: 0m 19s
[SAGE_INSTALL] SageAttention installation complete. Duration: 2m 51s

## 4090
[FLASH] Flash Attention installation complete. Duration: 3m 42s
[FLASH] Flash Attention installation complete. Duration: 0m 36s
[SAGE_BUILD] SageAttention build complete. Duration: 4m 32s
[SAGE_INSTALL] SageAttention installation complete. Duration: 0m 3s

--- download examples
[PROGRESS] 37.73GB / 37.73GB (100%) | Elapsed: 2m 5s | Speed: 309.09MB/s | ETA: 3m 55s (Setup)

-- download around 100Mb/s
[PROGRESS] 37.73GB / 37.73GB (100%) | Elapsed: 7m 47s | Speed: 82.73MB/s | ETA: 0m 0s
[PROGRESS] 37.73GB / 37.73GB (100%) | Elapsed: 8m 47s | Speed: 73.31MB/s | ETA: 0m 0s


++ instance creation time !!! 1m is acceptable and expected
total expected time - around 25 minutes for 4090 and 20m for 5090
