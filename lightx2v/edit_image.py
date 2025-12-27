import argparse
import os
import sys
import json
import random
import datetime
import uuid
from lightx2v import LightX2VPipeline

def get_unique_filename(original_path, output_dir):
    """Generates a unique filename based on the original image name."""
    base_name = os.path.basename(original_path)
    name, ext = os.path.splitext(base_name)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:6]
    return os.path.join(output_dir, f"{name}_edited_{timestamp}_{unique_id}{ext}")

def get_attn_mode():
    """Detects available attention mode."""
    try:
        import sageattention
        print("SageAttention detected. Using 'sage_attn'.")
        return "sage_attn"
    except ImportError:
        print("SageAttention not found. Falling back to 'flash_attn3'.")
        return "flash_attn3"

def main():
    parser = argparse.ArgumentParser(description="LightX2V Qwen Image Edit (I2I) with JSON Input")
    parser.add_argument("--json", type=str, required=True, help="Path to JSON input file")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save outputs")
    
    # Model Paths
    parser.add_argument("--model_path", type=str, default="/workspace/LightX2V/models/Qwen/Qwen-Image-Edit-2511", help="Path to Base Qwen Model")
    parser.add_argument("--quant_ckpt", type=str, default="/workspace/LightX2V/models/Qwen/Qwen-Image-Edit-2511-Lightning/qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning.safetensors", help="Path to Quantized Checkpoint")
    
    parser.add_argument("--steps", type=int, default=4, help="Inference steps")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: random)")

    args = parser.parse_args()

    # Handle Random Seed
    if args.seed is None:
        seed = random.randint(0, 2**32 - 1)
        print(f"No seed provided. Using random seed: {seed}")
    else:
        seed = args.seed
        print(f"Using provided seed: {seed}")

    # Parse JSON
    if not os.path.exists(args.json):
        print(f"Error: JSON file '{args.json}' not found.")
        sys.exit(1)
        
    with open(args.json, "r") as f:
        data = json.load(f)

    prompt = data.get("prompt")
    images = data.get("images", [])

    if not prompt or not images:
        print("Error: JSON must contain 'prompt' (string) and 'images' (list of strings).")
        sys.exit(1)

    print(f"Editing {len(images)} images with prompt: {prompt[:50]}...")

    # Initialize Pipeline
    pipe = LightX2VPipeline(
        model_path=args.model_path,
        model_cls="qwen-image-edit-2511",
        task="i2i",
    )

    if args.quant_ckpt and os.path.exists(args.quant_ckpt):
         print("Enabling quantization...")
         pipe.enable_quantize(
             dit_quantized=True, 
             dit_quantized_ckpt=args.quant_ckpt, 
             quant_scheme="fp8-sgl"
         )

    # Determine Attention Mode
    attn_mode = get_attn_mode()

    pipe.create_generator(
        attn_mode=attn_mode, 
        auto_resize=True,
        infer_steps=args.steps,
        guidance_scale=1,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    for i, img_path in enumerate(images):
        if not os.path.exists(img_path):
            print(f"Warning: Input image {img_path} not found. Skipping.")
            continue
            
        out_path = get_unique_filename(img_path, args.output_dir)
        
        print(f"Processing {img_path} -> {out_path}")
        
        pipe.generate(
            seed=seed,
            prompt=prompt,
            negative_prompt="",
            image_path=img_path,
            save_result_path=out_path
        )

    print("Batch processing complete.")

if __name__ == "__main__":
    main()
