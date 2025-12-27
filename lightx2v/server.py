from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import time
import os
from lightx2v import LightX2VPipeline
import contextlib

# Global pipeline variable
pipe = None

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global pipe
    print("[SERVER] Initializing LightX2V pipeline...")
    start_time = time.time()
    
    # Initialize pipeline
    pipe = LightX2VPipeline(
        model_path="/workspace/LightX2V/models/Qwen/Qwen-Image-Edit-2511",
        model_cls="qwen-image-edit-2511",
        task="i2i",
    )

    # Enable CPU offload
    pipe.enable_offload(
        cpu_offload=True,
        offload_granularity="block",
        text_encoder_offload=True,
        vae_offload=False,
    )
    
    # Create generator with torch_sdpa (fastest supported)
    pipe.create_generator(
        attn_mode="torch_sdpa",
        auto_resize=True,
        infer_steps=8,
        guidance_scale=1,
    )
    
    print(f"[SERVER] Model loaded in {time.time() - start_time:.2f}s")
    print("[SERVER] Ready to process requests!")
    yield
    print("[SERVER] Shutting down...")
    pipe = None

app = FastAPI(lifespan=lifespan)

class EditRequest(BaseModel):
    images: List[str]
    prompt: str
    negative_prompt: str = ""
    seed: int = 42

class EditResponse(BaseModel):
    status: str
    output_paths: List[str]
    generation_time: float

@app.post("/edit", response_model=EditResponse)
async def edit_image(request: EditRequest):
    if pipe is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    start_time = time.time()
    output_paths = []
    
    print(f"[Request] Processing {len(request.images)} images with prompt: '{request.prompt}'")
    
    try:
        for idx, img_path in enumerate(request.images):
            # Generate output path
            base, ext = os.path.splitext(img_path)
            out_path = f"{base}_edited_{int(time.time())}_{idx}{ext}"
            
            print(f"  - Processing: {img_path} -> {out_path}")
            
            pipe.generate(
                seed=request.seed,
                image_path=img_path,
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                save_result_path=out_path,
            )
            output_paths.append(out_path)
            
    except Exception as e:
        print(f"[Error] Generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
        
    duration = time.time() - start_time
    print(f"[Success] Completed in {duration:.2f}s")
    
    return EditResponse(
        status="success",
        output_paths=output_paths,
        generation_time=duration
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
