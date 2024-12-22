import os, argparse, json
from typing import List, Tuple
from collections import deque
import torch
from diffusers import DiffusionPipeline, EulerDiscreteScheduler, StableDiffusionPipeline
from transformers import Owlv2ForObjectDetection, Owlv2Processor
from utils.image_generator import generate_stable_diffusion
from utils.mask_generator import zero_shot_detection, generate_masks, visualize_boundings, visualize_masks
from utils.concept_impainter import impaint_concept
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def get_initial_tokens(prompt_token:str, prompt_eval:str)->Tuple[dict, dict]:
    '''
    Get the initial tokens which replace the special tokens by comparing the prompt with and without special tokens.
    Args:
        prompt_token (str): Prompt with special tokens.
        prompt_eval (str): Prompt without special tokens.
    Returns:
        Tuple[dict, dict]: A tuple of dictionaries containing the initial tokens and special tokens for objects and style.
    '''
    # use queues to store the words from the prompts
    prompt_token_words = deque(prompt_token.split())
    prompt_eval_words = deque(prompt_eval.split())
    # iterate through the words to find the initial tokens
    # temp variables
    target_token = None
    init_token = ""
    # result variables
    obj_init_tokens = []
    obj_special_tokens = []
    style_init_token = None
    style_special_token = None
    while prompt_token_words and prompt_eval_words:
        if prompt_token_words[0] == prompt_eval_words[0]: # words match, pass
            if target_token is not None: # save the initial token if it exists
                if prompt_token_words[0] == "style": # handle style tokens
                    style_init_token = init_token
                    style_special_token = target_token
                else: # handle object tokens
                    obj_init_tokens.append(init_token)
                    obj_special_tokens.append(target_token)
                target_token = None
                init_token = ""
            prompt_token_words.popleft()
            prompt_eval_words.popleft()
        elif target_token is None: # start of a token replacement
            target_token = prompt_token_words.popleft() # the special token
            init_token += prompt_eval_words.popleft() # the first word of initial token
        else:
            init_token += " " + prompt_eval_words.popleft() # continuation of initial token
    # save the last initial token if it exists
    if target_token is not None:
        obj_init_tokens.append(init_token)
        obj_special_tokens.append(target_token)
    # return the results
    obj_dict = {
        "init_tokens": obj_init_tokens,
        "special_tokens": obj_special_tokens,
    }
    style_dict = {
        "init_token": style_init_token,
        "special_token": style_special_token,
    }
    return obj_dict, style_dict

def read_prompts_json(json_path:str)->List[dict]:
    '''
    Read the JSON file containing prompts and extract the necessary information.
    Args:
        json_path (str): Path to the JSON file containing prompts.
    Returns:
        List[dict]: A list of dictionaries containing the necessary information for each prompt.
    '''
    with open(json_path, "r") as f:
        data = json.load(f)
        # extract information
        prompt_ids = list(data.keys())
        initial_prompts = []
        object_tokens = []
        for prompt_id in prompt_ids:
            # extract prompts
            prompt_token = data[prompt_id]["prompt"].rstrip('.').replace(">,", ">")
            initial_prompt = data[prompt_id]["prompt_4_clip_eval"].rstrip('.')
            # get the initial tokens
            obj_tok, sty_tok = get_initial_tokens(prompt_token, initial_prompt)
            # handle style prompt
            if sty_tok["init_token"] is not None:
                initial_prompt = initial_prompt.replace(sty_tok["init_token"], sty_tok["special_token"])
            # save the results
            initial_prompts.append(initial_prompt)
            object_tokens.append(obj_tok)

        # return the results
        custom_prompts = {
            "id": prompt_ids,
            "initial_prompts": initial_prompts,
            "object_tokens": object_tokens,
        }
    return custom_prompts

def parse_args():
    # arguments
    parser = argparse.ArgumentParser(description="Inference script for object detection.")
    parser.add_argument("--prompt", type=str, default="A dog and a cat.", help="List of prompts for image generation.")
    parser.add_argument("--special_tokens", nargs="+", type=str, default=["<dog>", "<cat>"], help="List of special tokens for textual inversion.")
    parser.add_argument("--init_tokens", nargs="+", type=str, default=["dog", "cat"], help="List of object tokens replacing the special tokens.")
    parser.add_argument("--image_per_prompt", type=int, default=1, help="Number of images to generate per prompt.")
    parser.add_argument("--json", type=str, default=None, help="Path to the JSON file containing prompts. This will override the prompts argument.")
    parser.add_argument("--inversion_dir", type=str, default=None, help="Path to the directory containing textual inversions.")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Path to the output directory.")
    parser.add_argument("--seed", type=int, default=1126, help="Random seed for reproducibility.")
    parser.add_argument("--init_steps", type=int, default=50, help="Number of steps for initial image generation.")
    parser.add_argument("--inpaint_steps", type=int, default=50, help="Number of steps for inpainting.")
    parser.add_argument("--inpaint_strength", type=float, default=1, help="Strength of inpainting.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for image generation.")
    parser.add_argument("--attn_slicing", action="store_true", help="Use attention slicing for image generation.")
    parser.add_argument("--precision", type=int, default=16, help="Floating point precision for image generation.")
    parser.add_argument("--width", type=int, default=512, help="Width of the generated images.")
    parser.add_argument("--height", type=int, default=512, help="Height of the generated images.")
    parser.add_argument("--save_process", action="store_true", help="Save the produced images (ex. masks) during the process.")
    parser.add_argument("--show_process", action="store_true", help="Show the produced images (ex. masks) during the process.")
    parser.add_argument("--model_type", type=str, default="sdxl", help="Type of model to use for inference. Options: 'sdxl' or 'sd2'.")
    parser.add_argument("--sd_model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0", help="Stable Diffusion model name or path.")
    parser.add_argument("--inpaint_model", type=str, default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1", help="Stable Diffusion inpainting model name or path.")
    args = parser.parse_args()
    # assertions
    assert len(args.special_tokens) == len(args.init_tokens), "[inference] Number of special tokens should match the number of initial tokens."
    assert args.image_per_prompt > 0, "[inference] Number of images per prompt should be greater than 0."
    # assign precision
    if args.precision == 16:
        args.dtype = torch.float16
    elif args.precision == 32:
        args.dtype = torch.float32
    else:
        print("[inference] Warning: Unusual precision value, defaulting to float16.")
        args.dtype = torch.float16
    # make output directory
    os.makedirs(args.output_dir, exist_ok=True)
    # handle prompts
    if args.json is not None: # read JSON file
        prompt_info = read_prompts_json(args.json)
    else: # use the provided prompts
        prompt_info = {
            "id": [0],
            "initial_prompts": [args.prompt],
            "object_tokens": [{"init_tokens": args.init_tokens, "special_tokens": args.special_tokens}]
        }
    # default image size
    if args.model_type == "sdxl":
        args.default_size = (1024, 1024)
    elif args.model_type == "sd2":
        args.default_size = (512, 512)
    else:
        print("[inference] Warning: Unusual model type, defaulting to SDXL.")
        args.default_size = (1024, 1024)
        args.model_type = "sdxl"

    return args, prompt_info

if __name__ == "__main__":
    ## Initialization
    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[inference] Device: {device}.")
    # parse arguments
    args, prompt_info = parse_args()
    torch.manual_seed(args.seed)
    prompt_ids = prompt_info["id"]
    initial_prompts = prompt_info["initial_prompts"]
    object_tokens = prompt_info["object_tokens"]
    print(f"[inference] Prompts: {initial_prompts}")
    print(f"[inference] Object tokens: {object_tokens}")

    ## Initial Image Generation
    # load diffusion model
    if args.model_type == "sdxl":
        print("[inference] Loading Stable Diffusion XL pipeline...")
        diffusion_pipeline = DiffusionPipeline.from_pretrained(args.sd_model, torch_dtype=args.dtype, use_safetensors=True, variant="fp16")
        print("[inference] Pipeline loaded successfully.")
        exit()
    else:
        print("[inference] Loading Stable Diffusion 2 pipeline...")
        diffusion_scheduler = EulerDiscreteScheduler.from_pretrained(args.sd_model, subfolder="scheduler")
        diffusion_pipeline = StableDiffusionPipeline.from_pretrained(args.sd_model, scheduler=diffusion_scheduler, torch_dtype=args.dtype)
        print("[inference] Pipeline loaded successfully.")
    # load textual inversions
    for inv_dir in os.listdir(args.inversion_dir):
        inv_path = os.path.join(args.inversion_dir, inv_dir)
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

    print("[inference] Loading Stable Diffusion pipeline...")
    diffusion_scheduler = EulerDiscreteScheduler.from_pretrained(args.sd_model, subfolder="scheduler")
    diffusion_pipeline = StableDiffusionPipeline.from_pretrained(args.sd_model, scheduler=diffusion_scheduler, torch_dtype=args.dtype)
    # load textual inversions
    inversion_dir = args.inversion_dir
    for inv_dir in os.listdir(inversion_dir):
        inv_path = os.path.join(inversion_dir, inv_dir)
        token_name = f"<{inv_dir}>"
        diffusion_pipeline.load_textual_inversion(inv_path, token=token_name)
    print("[inference] Pipeline loaded successfully.")
    # generate images
    init_images = generate_stable_diffusion(
        initial_prompts, model=args.sd_model, inversion_dir=args.inversion_dir, image_per_prompt=args.image_per_prompt,
        batch_size=args.batch_size, steps=args.init_steps, attention_slicing=args.attn_slicing,
        device=device, dtype=args.dtype)
    

    ## Post-Processing
    # load owlv2 model
    print("[inference] Loading object detection model...")
    detection_model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
    detection_processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    print("[inference] Object detection model loaded successfully.")
   
    # loop through the prompts
    for id, image_batch, object_tokens in zip(prompt_ids, init_images, object_tokens):
        # upack object tokens
        classes = object_tokens["init_tokens"]
        special_tokens = object_tokens["special_tokens"]
        # detect bounding boxes
        class_batch = [classes]*len(image_batch)
        bounding_batch = zero_shot_detection(image_batch, class_batch, processor=detection_processor, model=detection_model, device=device)
        # [showcase]: visualize results
        if args.show_process:
            visualize_boundings(image_batch, bounding_batch)
        # get the best bounding boxes for each class
        mask_bounding_batch = []
        for obj_classes, result in zip(class_batch, bounding_batch):
            bounding_boxes = []
            for obj_class in obj_classes:
                if result[obj_class]: # check if the list is not empty
                    bounding_boxes.append(result[obj_class].pop(0))
                else:
                    bounding_boxes.append(None)
            mask_bounding_batch.append(bounding_boxes)
        # generate mask images
        if args.sd_model == "stabilityai/stable-diffusion-2-1-base":
            size_tuple = (512, 512)
        elif args.sd_model == "stabilityai/stable-diffusion-xl-base-1.0":
            size_tuple = (1024, 1024)

        mask_batch = generate_masks(size_tuple, mask_bounding_batch)
        # save masks as images
        if args.save_process:
            # os.makedirs(os.path.join(args.output_dir, "masks", f"{id}"), exist_ok=True)
            # TODO: save masks as images
            pass
        # [showcase]: visualize masks
        if args.show_process:
            visualize_masks(mask_batch, image_batch, class_batch)



        # impanting concepts
        prompts = [[f"A {token}" for token in special_tokens] for _ in range(len(image_batch))]
        
        final_images = impaint_concept(
            image_batch, prompts, mask_batch, 
            model=args.inpaint_model, width=args.width, height=args.height, inversion_dir=inversion_dir, steps=args.inpaint_steps,
            attention_slicing=args.attn_slicing, strength=args.inpaint_strength,
            device=device, dtype=args.dtype)
        # save results
        os.makedirs(os.path.join(args.output_dir, f"{id}"), exist_ok=True)
        for i, image in enumerate(final_images):
            image.save(os.path.join(args.output_dir, f"{id}", f"{i}.png"))
        print(f"[inference] Prompt {id} completed.")
        # [showcase]: visualize final images
        if args.show_process:
            num_rows = np.ceil(np.sqrt(len(final_images))).astype(int)
            num_columns = np.ceil(len(final_images) / num_rows).astype(int)
            _, axes = plt.subplots(num_rows, num_columns, squeeze=False)
            axes = axes.flatten()
            for i, image in enumerate(final_images):
                axes[i].imshow(image)
                axes[i].axis("off")
            # turn off unused axes
            for ax in axes[len(final_images):]:
                ax.axis("off")
            plt.show()
    del detection_model, detection_processor
    torch.cuda.empty_cache()
    print("[inference] Inference completed.")