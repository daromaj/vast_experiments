# LongCat-Video Avatar-1.5 — Audio+Image → Video

Testing [LongCat-Video](https://github.com/meituan-longcat/LongCat-Video) + [Cache-DIT](https://github.com/vipshop/cache-dit) on rented GPU (Vast.ai).

**Target:** RTX 4090 (24 GB) / RTX 5090 (32 GB)  
**Strategy:** INT8 DiT + DMD 8-step distillation + Cache-DIT CPU offload

## Quick Start (Vast.ai)

```bash
# 1. Provisioning — run once, downloads ~45 GB models
bash longcat_video/provision_longcat.sh

# 2. Run inference
bash longcat_video/run_a2v.sh \
    --image /path/to/face.jpg \
    --audio /path/to/speech.wav \
    --resolution 480p
```

## Scripts

### `provision_longcat.sh`

Installs all dependencies and downloads model files. Idempotent — safe to re-run.

**What it installs:**
- System: aria2, ffmpeg, libgl, libsndfile
- Python: PyTorch 2.6.0+cu124, flash-attn 2.7.4, diffusers 0.35.1, cache-dit, sageattention
- Clones `meituan-longcat/LongCat-Video` repo

**What it downloads (selective, only needed files):**

| Component | Source | Size |
|-----------|--------|------|
| VAE | LongCat-Video base | 508 MB |
| Text encoder (UMT5-XXL) | LongCat-Video base | 22.7 GB |
| Tokenizer | LongCat-Video base | 21 MB |
| INT8 DiT (4 shards) | Avatar-1.5 `base_model_int8/` | 15.9 GB |
| DMD LoRA | Avatar-1.5 `lora/` | 2.5 GB |
| Scheduler config | Avatar-1.5 `scheduler/` | <1 MB |
| Vocal separator (ONNX) | Avatar-1.5 `vocal_separator/` | 67 MB |
| Whisper-large-v3 | Avatar-1.5 `whisper-large-v3/` (safetensors only) | ~3.1 GB |
| **Total** | | **~45 GB** |

Skipped: `dit/` from base model (54 GB, not needed for Avatar), `base_model/` fp32 from Avatar, whisper `.bin`/`.msgpack` duplicates.

### `run_a2v.sh`

Runs audio+image → video inference with INT8 + distillation.

```bash
# Minimal — 480p
./run_a2v.sh --image face.jpg --audio speech.wav

# 720p with custom prompt
./run_a2v.sh --image face.jpg --audio speech.wav \
    --resolution 720p \
    --prompt "A person giving a speech at a conference, professional lighting"

# Multi-GPU (if available)
./run_a2v.sh --image face.jpg --audio speech.wav --gpus 2

# Audio+text to video (no image, text prompt only)
./run_a2v.sh --image face.jpg --audio speech.wav --stage at2v
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--image` | required | Input face/character image |
| `--audio` | required | Speech audio (wav/mp3) |
| `--resolution` | `480p` | `480p` (480×832) or `720p` (768×1280) |
| `--prompt` | generic | Text prompt describing the scene |
| `--gpus` | `1` | Number of GPUs for torchrun |
| `--stage` | `ai2v` | `ai2v`=audio+image→video, `at2v`=audio+text→video |
| `--output-dir` | `./outputs_avatar` | Where to save output videos |

## VRAM Requirements

| GPU | 480p (INT8+distill) | 720p (INT8+distill) |
|-----|---------------------|---------------------|
| RTX 4090 (24 GB) | ✅ should fit | ⚠️ tight, may OOM |
| RTX 5090 (32 GB) | ✅ | ✅ should fit |

Cache-DIT bucket CPU offload trades ~5% speed for ~30% VRAM reduction.

## Directory Layout After Provisioning

```
/workspace/
├── LongCat-Video/              # cloned repo (Python scripts)
├── weights/
│   ├── LongCat-Video/          # base model components
│   │   ├── vae/                # diffusion_pytorch_model.safetensors (508 MB)
│   │   ├── text_encoder/       # UMT5-XXL shards (22.7 GB)
│   │   └── tokenizer/          # spiece model + configs
│   └── LongCat-Video-Avatar-1.5/
│       ├── config.json
│       ├── model_index.json
│       ├── base_model_int8/    # INT8 DiT shards (15.9 GB)
│       ├── lora/               # dmd_lora.safetensors (2.5 GB)
│       ├── scheduler/          # scheduler_config.json
│       ├── vocal_separator/    # Kim_Vocal_2.onnx (67 MB)
│       └── whisper-large-v3/   # model.safetensors + configs (~3.1 GB)
└── provisioning.log
```

## Quality / Speed Trade-offs

**Test results (2026-06-17, RTX 5090 32 GB):**
- INT8 + DMD 8-step → fatalna jakość. Video generuje się szybko (~2 min/segment) ale rezultat nieakceptowalny.
- Prawdopodobne przyczyny: za mało kroków denoisingu, kwantyzacja INT8, SDPA zamiast SageAttention.
- **SageAttention** (https://github.com/thu-ml/sageattention) — jest zainstalowane (v1.0.6). LongCat-Video wspiera przez `enable_flashattn3: true` (używa `flash_attn_interface` z SageAttention). **Nie użyliśmy go** — zamiast tego daliśmy SDPA fallback. Dla produkcji: włączyć flashattn3, będzie ~3× szybciej (~5s/step zamiast 13.5s).
- Dla RTX 5090 (Blackwell): provision skrypt buduje `sageattention3_blackwell` z optymalizacjami pod SM120.

**Warianty do przetestowania:**

| Wariant | VRAM | Czas/segment | Jakość |
|---------|------|-------------|--------|
| INT8 + DMD 8-step | 19 GB | ~2 min | ❌ fatalna |
| INT8 + więcej stepów | 19 GB | ~N×2 min | ? |
| FP16 + DMD 8-step | >32 GB | ~3 min | ? |
| FP16 + pełne 50-step | >32 GB | ~20 min | ✅ referencyjna |

**Kluczowe problemy rozwiązane podczas setupu:**
- `huggingface-cli` deprecated → używać `hf` CLI
- flash_attn kompiluje się 15+ min z source → warto dla produkcji (3× szybsze)
- CPU RAM OOM przy ładowaniu modelu (52 GB fp32 Lineary) → meta-device fix w `patch_quantization.py`
- VRAM: 19.2 GB peak na RTX 5090 (INT8 DiT 14.9 GB + VAE 0.5 GB + whisper 3.1 GB)
- `enable_flashattn2: true` w configu → wyłączyć jeśli nie ma flash_attn
- Brak SDPA fallback w attention modułach → dodany przez `patch_all_attention.py`
- DMD-LoRA ładuje 336 modułów
- `onnxruntime==1.16.3` nie ma kół dla Python 3.12 → `>=1.18.0`
- `tritonserverclient` i `libsndfile1` to fejkowe paczki w requirements_avatar.txt
- SageAttention: tylko wersja 1.0.6 istnieje (nie 2.2.0)

## Video Length

Domyślnie `--num-segments auto` (od wersji 2026-06-17) — skrypt `run_a2v.sh` sam oblicza liczbę segmentów z długości audio:
- 480p: 93 klatki @ 25 fps = 3.72s/segment, +3.2s za każdy kolejny
- 720p: 93 klatki @ 16 fps = 5.81s/segment, +5.0s za każdy kolejny

Dla 53s audio → 17 segmentów → ~30 min generacji (INT8+SDPA) lub ~11 min (z flash_attn).

## Manual Inference (without run_a2v.sh)

```bash
cd /workspace/LongCat-Video

# Create input JSON
cat > /tmp/input.json << 'EOF'
{
    "prompt": "A person speaking, facing the camera",
    "cond_image": "/path/to/image.jpg",
    "cond_audio": {"person1": "/path/to/audio.wav"}
}
EOF

# Run
torchrun --standalone --nnodes=1 --nproc_per_node=1 \
    run_demo_avatar_single_audio_to_video.py \
    --input_json /tmp/input.json \
    --output_dir ./outputs \
    --resolution 480p \
    --stage_1 ai2v \
    --checkpoint_dir /workspace/weights/LongCat-Video-Avatar-1.5 \
    --model_type avatar-v1.5 \
    --use_int8 \
    --use_distill \
    --num_segments 17
```
