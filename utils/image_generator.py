import os
from typing import List
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import matplotlib.pyplot as plt
import numpy as np

def generate_stable_diffusion(prompts:List[str], pipeline:any=None, 
                              steps:int=25, strength:float=7.5,
                              image_per_prompt:int=10, batch_size:int=1,
                              latents:torch.Tensor=None
                              ) -> List[List[Image.Image]]:
    """
    Generates images using a Stable Diffusion pipeline.
    
    Args:
        prompts (List[str]): A list of text prompts for image generation.
        pipeline (any, optional): A Stable Diffusion pipeline. If not provided, a default pipeline will be loaded. Default is None.
        steps (int, optional): Number of inference steps. Default is 25.
        strength (float, optional): Guidance scale for generation. Default is 7.5.
        image_per_prompt (int, optional): Number of images to generate per prompt. Default is 10.
        batch_size (int, optional): Batch size for inference. Default is 1.
        latent (torch.Tensor, optional): Latent tensor for generation. Default is None.
    Returns:
        List[List[Image.Image]]: A list of list of generated images for each prompt.
    """
    assert len(prompts) > 0, "[image generator] No prompts provided."
    # handle if no pipeline is provided
    if pipeline is None:
        print("[image generator] No pipeline provided. Loading default pipeline...")
        scheduler = EulerDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="scheduler")
        pipeline = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base", scheduler=scheduler, torch_dtype=torch.float16)
        print("[image generator] Pipeline loaded successfully.")
    # generate images
    results = []
    for prompt in prompts:
        images = []
        for i in range(np.ceil(image_per_prompt/batch_size).astype(int)):
            latent_batch = latents[i*batch_size:(i+1)*batch_size] if latents is not None else None
            latent_batch = torch.tensor(latent_batch) if latent_batch is not None else None
            image_batch = pipeline(
                prompt=prompt, 
                num_inference_steps=steps, guidance_scale=strength,
                num_images_per_prompt=batch_size, latent=latent_batch)
            images.extend(image_batch.images)
        results.append(images[:image_per_prompt])
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

     