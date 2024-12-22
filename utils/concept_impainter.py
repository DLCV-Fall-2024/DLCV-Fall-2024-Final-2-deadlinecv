import os
from typing import List, Tuple
import numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusionInpaintPipeline, AutoPipelineForInpainting

def impaint_concept(images:List[Image.Image], prompt_sets:List[List[str]], mask_sets:List[List[np.ndarray]],
                    model=None,
                    width=512, height=512,
                    inversion_dir=None,
                    steps:int=25, strength:float=1,
                    attention_slicing:bool=False, device:str="cuda",
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
    # assert len(textual_inversion_path) == len(textual_inversion_token), "[concept impainter] Number of textual inversion paths and tokens should be the same."
    # load model
    if model == "stabilityai/stable-diffusion-2-inpainting":
        print("[inference] Loading Stable Diffusion inpainting pipeline...")
        inpaint_pipeline = StableDiffusionInpaintPipeline.from_pretrained(model, torch_dtype=dtype)

        #load textual inversions
        print(f"[image generator] Loading {inversion_dir}")
        for inv_dir in os.listdir(inversion_dir):
            inv_path = os.path.join(inversion_dir, inv_dir)
            token_name = f"<{inv_dir}>"
            inpaint_pipeline.load_textual_inversion(inv_path, token=token_name)
        print("[inference] Pipeline loaded successfully.")

    elif model == "diffusers/stable-diffusion-xl-1.0-inpainting-0.1":
        inpaint_pipeline = AutoPipelineForInpainting.from_pretrained(model, torch_dtype=torch.float16, variant="fp16")
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
                        inpaint_pipeline.load_textual_inversion(safe_tensor_path, token=token_name, text_encoder=inpaint_pipeline.text_encoder_2, tokenizer=inpaint_pipeline.tokenizer_2)

                    elif "learned_embeds" in safe_tensor:
                        print(f"[image generator] Loading {safe_tensor_path} into text_encoder with token {token_name}")
                        inpaint_pipeline.load_textual_inversion(safe_tensor_path, token=token_name, text_encoder=inpaint_pipeline.text_encoder, tokenizer=inpaint_pipeline.tokenizer)

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
        inpaint_pipeline.enable_model_cpu_offload()
        inpaint_pipeline.enable_xformers_memory_efficient_attention()

    else:
        raise ValueError(
            f"{model} is not exists, please select between sd2 or sdxl."
        )
    
    # configure pipeline
    pipeline = inpaint_pipeline.to(device)
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
            image = image.resize((width, height), Image.LANCZOS) 
        results.append(image)
    del inpaint_pipeline
    torch.cuda.empty_cache()
    return results