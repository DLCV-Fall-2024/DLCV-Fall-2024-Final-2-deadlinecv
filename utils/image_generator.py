import os
from typing import List, Tuple
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler,  DiffusionPipeline
import matplotlib.pyplot as plt
import numpy as np

def generate_stable_diffusion(prompts:List[str], 
                              model=None, 
                              inversion_dir=None,
                     
                              steps:int=25, strength:float=7.5,
                              image_per_prompt:int=10, batch_size:int=1,
                              attention_slicing:bool=False, device:str="cuda",
                              dtype:torch.dtype=torch.float16
                              ) -> List[List[Image.Image]]:
    """
    Generates images using a Stable Diffusion pipeline.
    
    Args:
        prompts (List[str]): A list of text prompts for image generation.
        pipeline (StableDiffusionPipeline, optional): Preloaded Stable Diffusion pipeline. If None, stabilityai/stable-diffusion-2-1-base will be loaded.
        textual_inversion_path (List[str], optional): List of paths to textual inversion embeddings.
        textual_inversion_token (List[str], optional): List of tokens for textual inversion.
        # image_size (Tuple[int, int], optional): Size of the generated images (width, height). Default is (512, 512).
        steps (int, optional): Number of inference steps. Default is 25.
        strength (float, optional): Guidance scale for generation. Default is 7.5.
        attention_slicing (bool, optional): Whether to enable attention slicing to reduce VRAM usage at the cost of speed. Default is True.
        device (str, optional): Device to run inference on ('cuda' or 'cpu'). Default is 'cuda'.
        batch_size (int, optional): Number of prompts to process at once. Default is 1.
        dtype (torch.dtype, optional): Data type for the pipeline. Will be ignored if you pass the pipeline argument. Default is torch.float16. Some GPUs do not support torch.float16, you will see "RuntimeWarning: invalid value encountered in cast
  images = (images * 255).round().astype("uint8")" and get a black image. Change to torch.float32 if you encounter this issue.
    Returns:
        List[List[Image.Image]]: A list of list of generated images for each prompt.
    """
    assert len(prompts) > 0, "[image generator] No prompts provided."
    # assert len(textual_inversion_path) == len(textual_inversion_token), "[image generator] Number of textual inversion paths and tokens should be the same."
    # load model 
    print(f"Before Loading Pipeline ... Allocated Memory: {torch.cuda.memory_allocated() / 1e6} MB")
    print("[inference] Loading Stable Diffusion pipeline...")
    
    if model == "stabilityai/stable-diffusion-2-1-base":
        diffusion_scheduler = EulerDiscreteScheduler.from_pretrained(model, subfolder="scheduler")
        diffusion_pipeline = StableDiffusionPipeline.from_pretrained(model, scheduler=diffusion_scheduler, torch_dtype=dtype)
        print("[inference] Pipeline loaded successfully.")

        #load textual inversions
        print(f"[image generator] Loading {inversion_dir}")
        for inv_dir in os.listdir(inversion_dir):
            inv_path = os.path.join(inversion_dir, inv_dir)
            token_name = f"<{inv_dir}>"
            diffusion_pipeline.load_textual_inversion(inv_path, token=token_name)
    
    elif model == "stabilityai/stable-diffusion-xl-base-1.0":
        diffusion_pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=dtype, use_safetensors=True, variant="fp16")
        print("[inference] Pipeline loaded successfully.")
    
        #load textual inversions
        for inv_dir in os.listdir(inversion_dir):
            inv_path = os.path.join(inversion_dir, inv_dir)
            token_name = f"<{inv_dir}>"
            for safe_tensor in os.listdir(inv_path):
                safe_tensor_path = os.path.join(inv_path, safe_tensor)

                if safe_tensor.endswith(".safetensors"):
                    
                    if "learned_embeds_2" in safe_tensor:
                        print(f"[image generator] Loading {safe_tensor_path} into text_encoder 2 with token {token_name}")
                        diffusion_pipeline.load_textual_inversion(safe_tensor_path, token=token_name, text_encoder=diffusion_pipeline.text_encoder_2, tokenizer=diffusion_pipeline.tokenizer_2)

                    elif "learned_embeds" in safe_tensor:
                        print(f"[image generator] Loading {safe_tensor_path} into text_encoder with token {token_name}")
                        diffusion_pipeline.load_textual_inversion(safe_tensor_path, token=token_name, text_encoder=diffusion_pipeline.text_encoder, tokenizer=diffusion_pipeline.tokenizer)

                    else:
                        raise ValueError(
                            f"[image generator] Unexpected safetensor file format: {safe_tensor_path}. "
                            f"Expected filenames containing 'learned_embeds' or 'learned_embeds_2'."
                        )
                    
                else:
                    raise ValueError(
                    f"[image generator] Unsupported file format: {safe_tensor_path}. "
                    f"Only .safetensors files are allowed."
                )

    else:
        raise ValueError(
            f"{model} is not exists, please select between sd2 or sdxl."
        )

    # configure pipeline
    pipeline = diffusion_pipeline.to(device)
    if attention_slicing:
        pipeline.enable_attention_slicing()
    # generate images
    results = []
    for prompt in prompts:
        images = []
        for _ in range(np.ceil(image_per_prompt/batch_size).astype(int)):
            print(f"After Loading Pipeline ... Allocated Memory: {torch.cuda.memory_allocated() / 1e6} MB")
            image_batch = pipeline(prompt=prompt, 
                              num_inference_steps=steps, guidance_scale=strength,
                              num_images_per_prompt=batch_size)

            images.extend(image_batch.images)
        results.append(images[:image_per_prompt])

    # clean up memory
    if model == "stabilityai/stable-diffusion-2-1-base":
        del diffusion_pipeline, diffusion_scheduler
    elif model == "stabilityai/stable-diffusion-xl-base-1.0":
        del diffusion_pipeline
    torch.cuda.empty_cache()
    print("[inference] Initial images generated successfully, memory released.")
    return results
    
# example usage
if __name__ == "__main__":
    # directory
    code_dir = os.path.dirname(os.path.abspath(__file__))
    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}.")
    torch.manual_seed(42)
    # generate images
    prompts = ["A cat wearing wearable glasses in a <watercolor> style."]
    textual_inversion_dir = os.path.join(code_dir, os.pardir, "textual_inversions", "sd2")
    inversion_paths = [os.path.join(textual_inversion_dir, inv) for inv in ["watercolor"]]
    inversion_token = ["<watercolor>"]
    images = generate_stable_diffusion(
        prompts, image_per_prompt=1,
        textual_inversion_path=inversion_paths,
        textual_inversion_token=inversion_token,
        attention_slicing=False, device=device,
        batch_size=1, steps=10, dtype=torch.float32)
    images = images[0]
    # visualize pil images
    num_rows = np.ceil(np.sqrt(len(images))).astype(int)
    num_columns = np.ceil(len(images) / num_rows).astype(int)
    _, axes = plt.subplots(num_rows, num_columns, squeeze=False)
    axes = axes.flatten()
    for i, image in enumerate(images):
        axes[i].imshow(image)
        axes[i].axis("off")
    # turn off unused axes
    for ax in axes[len(images):]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()

     