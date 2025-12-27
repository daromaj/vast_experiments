import argparse
import os
import sys
import random
import time
import uuid
import datetime
from lightx2v import LightX2VPipeline

def get_unique_filename(base_path):
    """Generates a unique filename by appending timestamp and short UUID."""
    if not base_path:
        base_path = "output.png"
    
    name, ext = os.path.splitext(base_path)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:6]
    return f"{name}_{timestamp}_{unique_id}{ext}"

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
    parser = argparse.ArgumentParser(description="LightX2V Qwen Image Generation (T2I)")
    parser.add_argument("--prompt", type=str, help="Text prompt for generation")
    parser.add_argument("--prompt_file", type=str, help="Path to file containing the prompt")
    parser.add_argument("--output", type=str, default="output.png", help="Base path for the generated image (will be uniqueified)")
    parser.add_argument("--model_path", type=str, default="/workspace/LightX2V/models/Qwen/Qwen-Image-Edit-2511-Lightning", help="Path to Qwen model")
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

    # Handle Prompt Input
    if args.prompt:
        prompt = args.prompt
    elif args.prompt_file:
        if not os.path.exists(args.prompt_file):
            print(f"Error: Prompt file '{args.prompt_file}' not found.")
            sys.exit(1)
        with open(args.prompt_file, "r") as f:
            prompt = f.read().strip()
    else:
        print("Error: Must provide either --prompt or --prompt_file")
        sys.exit(1)

    # Generate unique output path
    output_path = get_unique_filename(args.output)
    
    print(f"Generating image...")
    print(f"Prompt: {prompt[:100]}...")
    
    # Initialize Pipeline
    pipe = LightX2VPipeline(
        model_path=args.model_path,
        model_cls="qwen-image-edit-2511", # Using the Edit model in T2I mode
        task="t2i", 
    )

    # Enable quantization if Lightning model
    if "Lightning" in args.model_path:
        ckpt_path = os.path.join(args.model_path, "Qwen-quant/qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning.safetensors")
        if os.path.exists(ckpt_path):
             pipe.enable_quantize(
                 dit_quantized=True, 
                 dit_quantized_ckpt=ckpt_path, 
                 quant_scheme="fp8-sgl"
             )
        else:
            print(f"Warning: Quantized checkpoint not found at {ckpt_path}. Running without explicit quantization config.")

    # Determine Attention Mode
    attn_mode = get_attn_mode()

    # Create Generator
    pipe.create_generator(
        attn_mode=attn_mode,
        auto_resize=True,
        infer_steps=args.steps,
        guidance_scale=1, 
    )

    # Generate
    pipe.generate(
        seed=seed,
        prompt=prompt,
        negative_prompt="", 
        save_result_path=output_path,
    )
    
    print(f"Success! Image saved to {output_path}")

if __name__ == "__main__":
    main()
