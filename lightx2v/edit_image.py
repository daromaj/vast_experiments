"""
Qwen-Image-Edit-2511 I2I (Image-to-Image) Script
Optimized for 32GB GPU with CPU offload
"""

import argparse
import os
import json
from lightx2v import LightX2VPipeline

def main():
    parser = argparse.ArgumentParser(description="Qwen Image Edit (I2I) with JSON Input")
    parser.add_argument("--json", type=str, required=True, help="Path to JSON input file")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save outputs")
    parser.add_argument("--model_path", type=str, 
                       default="/workspace/LightX2V/models/Qwen/Qwen-Image-Edit-2511",
                       help="Path to Qwen-Image-Edit-2511 model")
    parser.add_argument("--steps", type=int, default=8, help="Inference steps (default: 8)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    # Parse JSON input
    if not os.path.exists(args.json):
        print(f"Error: JSON file '{args.json}' not found.")
        return
    
    with open(args.json, "r") as f:
        data = json.load(f)
    
    # Expect array of images
    images = data.get("images", [])
    prompt = data.get("prompt", "")
    negative_prompt = data.get("negative_prompt", "")
    
    if not images or not isinstance(images, list):
        print("Error: JSON must contain 'images' (array of image paths)")
        return
    
    if not prompt:
        print("Error: JSON must contain 'prompt'")
        return
    
    if not prompt:
        print("Error: JSON must contain 'prompt'")
        return
    
    print(f"Processing {len(images)} image(s) with prompt: '{prompt}'")
    
    # Initialize pipeline with CPU offload
    print("Initializing pipeline with CPU offload...")
    pipe = LightX2VPipeline(
        model_path=args.model_path,
        model_cls="qwen-image-edit-2511",
        task="i2i",
    )
    
    # Enable CPU offload to fit in 32GB VRAM
    pipe.enable_offload(
        cpu_offload=True,
        offload_granularity="block",
        text_encoder_offload=True,
        vae_offload=False,
    )
    
    # Create generator
    pipe.create_generator(
        attn_mode="torch_sdpa",  # Standard PyTorch attention
        auto_resize=True,
        infer_steps=args.steps,
        guidance_scale=1,
    )
    
    # Process each image
    os.makedirs(args.output_dir, exist_ok=True)
    
    for idx, image_path in enumerate(images):
        if not os.path.exists(image_path):
            print(f"Warning: Image '{image_path}' not found, skipping...")
            continue
        
        # Generate output filename
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        output_path = os.path.join(args.output_dir, f"{name}_edited{ext}")
        
        print(f"\n[{idx+1}/{len(images)}] Processing: {image_path}")
        print(f"  Output: {output_path}")
        
        # Generate
        pipe.generate(
            seed=args.seed,
            image_path=image_path,
            prompt=prompt,
            negative_prompt=negative_prompt,
            save_result_path=output_path,
        )
        
        print(f"  ✅ Saved to: {output_path}")
    
    print(f"\n✅ Completed! Processed {len(images)} image(s)")

if __name__ == "__main__":
    main()
