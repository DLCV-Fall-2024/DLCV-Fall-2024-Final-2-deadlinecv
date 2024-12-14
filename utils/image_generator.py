from typing import List, Tuple
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import matplotlib.pyplot as plt
import numpy as np

def generate_stable_diffusion(prompts:List[str], pipeline:StableDiffusionPipeline=None,
                              textual_inversion_path:List[str]=[],
                              image_size:Tuple[int]=(512, 512),
                              steps:int=25, strength:float=7.5,
                              image_per_prompt:int=2, batch_size:int=2,
                              attention_slicing:bool=True, device:str="cuda",
                              dtype:torch.dtype=torch.float16
                              ) -> List[List[Image.Image]]:
    """
    Generates images using a Stable Diffusion pipeline.
    
    Args:
        prompts (List[str]): A list of text prompts for image generation.
        pipeline (StableDiffusionPipeline, optional): Preloaded Stable Diffusion pipeline. If None, stabilityai/stable-diffusion-2-1-base will be loaded.
        textual_inversion_path (List[str], optional): List of paths to textual inversion embeddings.
        image_size (Tuple[int, int], optional): Size of the generated images (width, height). Default is (512, 512).
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
    # load model
    if pipeline is None:
        print("[image generator] No pipeline provided. Loading default pipeline...")
        scheduler = EulerDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="scheduler")
        pipeline = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base", scheduler=scheduler, torch_dtype=dtype)
        print("[image generator] Pipeline loaded successfully.")
    # load textual inversions
    for inversion_path in textual_inversion_path:
        pipeline.load_textual_inversion(inversion_path)
    # configure pipeline
    pipeline = pipeline.to(device)
    if attention_slicing:
        pipeline.enable_attention_slicing()
    # generate images
    results = []
    for prompt in prompts:
        images = []
        for _ in range(np.ceil(image_per_prompt/batch_size).astype(int)):
            image_batch = pipeline(prompt=prompt, width=image_size[0], height=image_size[1], 
                              num_inference_steps=steps, guidance_scale=strength,
                              num_images_per_prompt=batch_size)
            images.extend(image_batch.images)
        results.append(images[:image_per_prompt])
    return results
    
# example usage
if __name__ == "__main__":
    prompts = ["A cat on the right and a dog on the left."]
    images = generate_stable_diffusion(prompts, image_per_prompt=4, batch_size=2, steps=25, dtype=torch.float32)
    images = images[0]
    # visualize pil images
    if len(images) == 1:
        plt.figure()
        plt.imshow(images[0])
        plt.axis("off")
        plt.show()
    else:
        num_rows = np.ceil(np.sqrt(len(images))).astype(int)
        num_columns = np.ceil(len(images) / num_rows).astype(int)
        _, axes = plt.subplots(num_rows, num_columns)
        axes = axes.flatten()
        for i, image in enumerate(images):
            axes[i].imshow(image)
            axes[i].axis("off")
        # turn off unused axes
        for ax in axes[len(images):]:
            ax.axis("off")
        plt.tight_layout()
        plt.show()

     