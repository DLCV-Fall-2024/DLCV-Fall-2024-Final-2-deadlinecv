from typing import List, Tuple
import numpy as np
from PIL import Image

def impaint_concept(images:List[Image.Image], prompt_sets:List[List[str]], mask_sets:List[List[np.ndarray]],
                    pipeline:any=None,
                    size:Tuple[int, int]=(512, 512),
                    steps:int=25, strength:float=1
                    ) -> List[Image.Image]:
    '''
    Impaints concepts into images using a Stable Diffusion inpainting pipeline.
    Args:
        images (List[Image.Image]): A list of images to inpaint.
        prompt_sets (List[List[str]]): A list of list of text prompts for inpainting.
        mask_sets (List[List[np.ndarray]]): A list of list of masks for inpainting.
        pipeline (StableDiffusionInpaintPipeline, optional): Preloaded Stable Diffusion inpainting pipeline. If None, stabilityai/stable-diffusion-2-inpainting will be loaded.
        steps (int, optional): Number of inference steps. Default is 25.
        strength (float, optional): Guidance scale for generation. Default is 1.
    Returns:
        List[Image.Image]: A list of inpainted images.
    '''
    assert len(images) == len(prompt_sets) == len(prompt_sets), "[concept impainter] Number of images, prompts, and masks should be the same."
    # impaint concepts
    results = []
    for image, prompt_set, mask_set in zip(images, prompt_sets, mask_sets):
        assert len(prompt_set) == len(mask_set), "[concept impainter] Number of prompts and masks should be the same."
        for prompt, mask in zip(prompt_set, mask_set):
            if mask is None:
                continue
            image = pipeline(prompt=prompt, image=image, mask_image=mask,
                             width=image.width, height=image.height,
                             num_inference_steps=steps, strength=strength).images[0]
            image = image.resize(size, Image.LANCZOS) 
        results.append(image)
    return results