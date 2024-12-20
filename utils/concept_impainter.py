import os
from typing import List, Tuple
import numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusionInpaintPipeline

def impaint_concept(images:List[Image.Image], prompt_sets:List[List[str]], mask_sets:List[List[np.ndarray]],
                    pipeline:StableDiffusionInpaintPipeline=None,
                    textual_inversion_path:List[str]=[],
                    textual_inversion_token:List[str]=[],
                    steps:int=25, strength:float=1,
                    attention_slicing:bool=True, device:str="cuda",
                    dtype:torch.dtype=torch.float16
                    ) -> List[Image.Image]:
    '''
    Impaints concepts into images using a Stable Diffusion inpainting pipeline.
    Args:
        images (List[Image.Image]): A list of images to inpaint.
        prompt_sets (List[List[str]]): A list of list of text prompts for inpainting.
        mask_sets (List[List[np.ndarray]]): A list of list of masks for inpainting.
        pipeline (StableDiffusionInpaintPipeline, optional): Preloaded Stable Diffusion inpainting pipeline. If None, stabilityai/stable-diffusion-2-inpainting will be loaded.
        textual_inversion_path (List[str], optional): List of paths to textual inversion embeddings. If you pass the model in, you can load the textual inversions yourself and ignore this argument.
        textual_inversion_token (List[str], optional): List of tokens for textual inversion. Same as above.
        steps (int, optional): Number of inference steps. Default is 25.
        strength (float, optional): Guidance scale for generation. Default is 1.
        attention_slicing (bool, optional): Whether to enable attention slicing to reduce VRAM usage at the cost of speed. Default is True.
        device (str, optional): Device to run inference on ('cuda' or 'cpu'). Default is 'cuda'.
        dtype (torch.dtype, optional): Data type for the pipeline. Will be ignored if you pass the pipeline argument. Default is torch.float16. Some GPUs do not support torch.float16, you will see "RuntimeWarning: invalid value encountered in cast
    images = (images * 255).round().astype("uint8")" and get a black image. Change to torch.float32 if you encounter this issue.
    Returns:
        List[Image.Image]: A list of inpainted images.
    '''
    assert len(images) == len(prompt_sets) == len(prompt_sets), "[concept impainter] Number of images, prompts, and masks should be the same."
    assert len(textual_inversion_path) == len(textual_inversion_token), "[concept impainter] Number of textual inversion paths and tokens should be the same."
    # load model
    if pipeline is None:
        print("[concept impainter] No pipeline provided. Loading default pipeline...")
        pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=dtype
        )
        print("[concept impainter] Pipeline loaded successfully.")
    # load textual inversions
    for inversion_path, inversion_token in zip(textual_inversion_path, textual_inversion_token):
        pipeline.load_textual_inversion(inversion_path, token=inversion_token)
    # configure pipeline
    pipeline = pipeline.to(device)
    if attention_slicing:
        pipeline.enable_attention_slicing()
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
        results.append(image)
    return results