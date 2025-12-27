# LightX2V Qwen-Image-Edit Server

#TODO
test https://lightx2v-en.readthedocs.io/en/latest/deploy_guides/deploy_gradio.html


A high-performance, persistent image editing server using **Qwen-Image-Edit-2511** (Full Model) optimized for **32GB VRAM GPUs** via CPU offloading.

## Quickstart

### 1. Provision on Vast.ai
Use the provided provisioning script on a compatible GPU instance (min 32GB VRAM, e.g., RTX 3090/4090/5090).

```bash
# Run this on the Vast.ai instance
bash provision_lightx2v_qwen.sh
```
*Note: This script installs dependencies, downloads the ~55GB model, and automatically starts the API server in the background.*

### 2. Usage
The server runs on port `8000`. You can interact with it using the provided helper script or direct HTTP requests.

**Using Helper Script:**
```bash
# Syntax: ./example_server_request.sh [image_path] [prompt]
/workspace/example_server_request.sh /workspace/test.jpg "make it sunset"
```

**Using Curl:**
```bash
curl -X POST "http://localhost:8000/edit" \
     -H "Content-Type: application/json" \
     -d '{
           "images": ["/workspace/image.jpg"],
           "prompt": "add fireworks in the sky",
           "seed": 42
         }'
```

---

## Technical Details

### Architecture
- **Model:** [Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) (Full BFloat16 version)
- **Framework:** LightX2V + PyTorch
- **Optimization:** CPU Offloading (Text Encoder offloaded to RAM to fit 32GB VRAM)
- **Attention:** `torch_sdpa` (Standard PyTorch Scaled Dot Product Attention)
- **Server:** FastAPI + Uvicorn (Persistent model loading)

### Performance Metrics (RTX 5090 / 32GB VRAM)
| Method | Initialization | Generation (8 steps) | Total Latency |
| :--- | :--- | :--- | :--- |
| **Cold Script** (`edit_image.py`) | ~35.5s | ~21.5s | **~57.0s** |
| **Persistent Server** (`server.py`) | 0s (Pre-loaded) | ~20.0s | **~20.0s** |

*Note: The server eliminates the heavy model loading cost for every request, resulting in ~3x faster response times.*

### Key Files
- **`provision_lightx2v_qwen.sh`**: Setup script. Installs system/python dependencies, downloads the ~55GB model, installs `fastapi/uvicorn`, and starts `server.py`.
- **`server.py`**: FastAPI application. Initializes the LightX2V pipeline with CPU offload at startup and exposes the `/edit` endpoint.
- **`edit_image.py`**: Standalone Python script for single-run editing (useful for debugging or batch processing without a server).
- **`example_server_request.sh`**: Simple bash script to send curl requests to the local server.

### Hardware Requirements
- **GPU:** NVIDIA GPU with **≥32GB VRAM** (Required for running the full model with offloading)
- **RAM:** ≥64GB Recommended (to hold offloaded weights)
- **Disk:** ≥100GB (Model is ~58GB, plus dependencies and system overhead)
