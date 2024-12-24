import os, argparse, json
from typing import List, Tuple
from collections import deque
import torch
from diffusers import DiffusionPipeline, EulerDiscreteScheduler, StableDiffusionPipeline, StableDiffusionInpaintPipeline, AutoPipelineForInpainting
from transformers import Owlv2ForObjectDetection, Owlv2Processor
from utils.image_generator import generate_stable_diffusion
from utils.mask_generator import zero_shot_detection, generate_masks, visualize_boundings, visualize_masks
from utils.concept_impainter import impaint_concept
import numpy as np
import matplotlib.pyplot as plt

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

def read_prompts_json(json_path:str, token_annotation:dict)->List[dict]:
    '''
    Read the JSON file containing prompts and extract the necessary information.
    Args:
        json_path (str): Path to the JSON file containing prompts.
        token_annotation (dict): A dictionary containing the annotations for special tokens.
    Returns:
        List[dict]: A list of dictionaries containing the necessary information for each prompt.
    '''
    with open(json_path, "r") as f:
        data = json.load(f)
        # extract information
        prompt_ids = list(data.keys())
        initial_prompts = []
        object_tokens = []
        style_tokens = []
        for prompt_id in prompt_ids:
            # extract prompts
            token_prompt = data[prompt_id]["prompt"].rstrip('.').replace(">,", ">")
            special_tokens = data[prompt_id]["token_name"]
            # get the initial tokens
            obj_init_tokens, obj_special_tokens, obj_id_tokens = [], [], []
            style_special_token = None
            for token in special_tokens:
                if token in token_annotation["object"]:
                    obj_init_tokens.append(token_annotation["object"][token]["init_token"])
                    obj_id_tokens.append(token_annotation["object"][token]["id_token"])
                    obj_special_tokens.append(token)
                elif token in token_annotation["style"]:
                    style_special_token = token
            # replace the object special tokens with initial tokens
            initial_prompt = token_prompt
            for init_token, special_token in zip(obj_init_tokens, obj_special_tokens):
                initial_prompt = initial_prompt.replace(special_token, init_token)
            # save the results
            initial_prompts.append(initial_prompt)
            object_tokens.append({"init_tokens": obj_init_tokens, "special_tokens": obj_special_tokens, "id_tokens": obj_id_tokens})
            style_tokens.append(style_special_token)
        # return the results
        custom_prompts = {
            "id": prompt_ids,
            "initial_prompts": initial_prompts,
            "object_tokens": object_tokens,
            "style_tokens": style_tokens
        }
    return custom_prompts

def load_latents(latent_dir:str, num:int, image_shape:Tuple[int, int])->torch.Tensor:
    '''
    Read the first num latent tensors from the latent directory.
    Args:
        latent_dir (str): Path to the latent tensor.
        num (int): Number of latent tensors to read.
        image_shape (Tuple[int, int]): Shape of the image to generate.
    Returns:
        torch.Tensor: The latent tensor.
    '''
    latent_height, latent_width = image_shape[0] // 8, image_shape[1] // 8
    latent_shape = (4, latent_height, latent_width) # 4 channels
    latent_list = []
    for i in range(num):
        if latent_dir is None:
            latent = torch.randn(latent_shape)
        else:
            latent_path = os.path.join(latent_dir, f"{i}.pt")
            if not os.path.exists(latent_path):
                print(f"[inference] Warning: Latent tensor not enough, generating random latents.")
                latent = torch.randn(latent_shape)
            else:
                latent = torch.load(latent_path)
        latent_list.append(latent)

def parse_args():
    # arguments
    parser = argparse.ArgumentParser(description="Inference script for object detection.")
    parser.add_argument("--prompt", type=str, default="A <cat2> on the right and a <dog6> on the left.", help="List of prompts for image generation.")
    parser.add_argument("--special_tokens", nargs="+", type=str, default=["<dog6>", "<cat2>"], help="List of special tokens for textual inversion.")
    parser.add_argument("--init_tokens", nargs="+", type=str, default=["corgi", "grey cat"], help="List of object tokens replacing the special tokens.")
    parser.add_argument("--id_tokens", nargs="+", type=str, default=["dog", "cat"], help="List of tags for the detection.")
    parser.add_argument("--style_special_token", type=str, default=None, help="Special token for style inversion.")
    parser.add_argument("--image_per_prompt", type=int, default=1, help="Number of images to generate per prompt.")
    parser.add_argument("--json", type=str, default=None, help="Path to the JSON file containing prompts. This will override the prompts argument.")
    parser.add_argument("--inversion_dir", type=str, default=None, help="Path to the directory containing textual inversions.")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Path to the output directory.")
    parser.add_argument("--seed", type=int, default=1126, help="Random seed for reproducibility.")
    parser.add_argument("--init_steps", type=int, default=25, help="Number of steps for initial image generation.")
    parser.add_argument("--inpaint_steps", type=int, default=25, help="Number of steps for inpainting.")
    parser.add_argument("--inpaint_strength", type=float, default=0.7, help="Strength of inpainting.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for image generation.")
    parser.add_argument("--attn_slicing", action="store_true", help="Use attention slicing for image generation.")
    parser.add_argument("--xformer", action="store_true", help="Use memory efficient attention for image generation.")
    parser.add_argument("--precision", type=int, default=16, help="Floating point precision for image generation.")
    parser.add_argument("--width", type=int, default=512, help="Width of the generated images.")
    parser.add_argument("--height", type=int, default=512, help="Height of the generated images.")
    parser.add_argument("--save_process", action="store_true", help="Save the produced images (ex. masks) during the process.")
    parser.add_argument("--show_process", action="store_true", help="Show the produced images (ex. masks) during the process.")
    parser.add_argument("--model_type", type=str, default="sdxl", help="Type of model to use for inference. Options: 'sdxl' or 'sd2'.")
    parser.add_argument("--sd_model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0", help="Stable Diffusion model name or path.")
    parser.add_argument("--inpaint_model", type=str, default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1", help="Stable Diffusion inpainting model name or path.")
    parser.add_argument("--mask_padding", type=int, default=None, help="Inpainting with mask padding.")
    parser.add_argument("--init_latent_dir", type=str, default=None, help="Path to the latent tensor for image generation.")
    args = parser.parse_args()
    # assertions
    assert len(args.special_tokens) == len(args.init_tokens), "[inference] Number of special tokens should match the number of initial tokens."
    assert args.image_per_prompt > 0, "[inference] Number of images per prompt should be greater than 0."
    args.image_per_prompt = np.ceil(args.image_per_prompt/args.batch_size).astype(int) * args.batch_size # make it divisible by batch size
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
    # read token annotations
    if args.inversion_dir is not None:
        # assert there is a annotation.json file in args.inversion_dir
        assert "annotation.json" in os.listdir(args.inversion_dir), "[inference] annotation.json not found in the inversion directory."
        with open(os.path.join(args.inversion_dir, "annotation.json"), "r") as f:
            token_annotation = json.load(f)
    else:
        token_annotation = {}
    # default image size
    if args.model_type == "sdxl":
        args.default_size = (1024, 1024)
    elif args.model_type == "sd2":
        args.default_size = (512, 512)
    else:
        print("[inference] Warning: Unusual model type, defaulting to SDXL.")
        args.default_size = (1024, 1024)
        args.model_type = "sdxl"
    # handle id tokens
    if len(args.id_tokens) != len(args.init_tokens):
        print("[inference] Warning: Number of id tokens should match the number of initial tokens, using the initial tokens as id tokens.")
        args.id_tokens = args.init_tokens
    # handle prompts
    if args.json is not None: # read JSON file
        prompt_info = read_prompts_json(args.json, token_annotation)
    else: # use the provided prompts
        initial_prompt = args.prompt.rstrip('.').replace(">,", ">")
        for special_token, init_token in zip(args.special_tokens, args.init_tokens):
            initial_prompt = initial_prompt.replace(special_token, init_token)
        prompt_info = {
            "id": [0],
            "initial_prompts": [initial_prompt],
            "object_tokens": [{"init_tokens": args.init_tokens, "special_tokens": args.special_tokens, "id_tokens": args.id_tokens}],
            "style_tokens": [args.style_special_token],
            "init_latents": [load_latents(args.init_latent_dir, args.image_per_prompt, args.default_size)]
        }
    

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
    style_tokens = prompt_info["style_tokens"]
    init_latents = prompt_info["init_latents"]
    print(f"[inference] Prompts: {initial_prompts}")
    print(f"[inference] Object tokens: {object_tokens}")
    print(f"[inference] Style tokens: {style_tokens}")
    # save latents for reproducibility
    if args.save_process:
        for id, latent in zip(prompt_ids, init_latents):
            os.makedirs(os.path.join(args.output_dir, "initial_latents", f"{id}"), exist_ok=True)
            for i, latent_tensor in enumerate(latent):
                torch.save(latent_tensor, os.path.join(args.output_dir, "initial_latents", f"{id}", f"{i}.pt"))
    exit()
    ## Initial Image Generation
    # load diffusion model
    if args.model_type == "sdxl":
        print("[inference] Loading Stable Diffusion XL pipeline...")
        diffusion_pipeline = DiffusionPipeline.from_pretrained(args.sd_model, torch_dtype=args.dtype, use_safetensors=True, variant="fp16")
        # load textural inversions
        for inv_dir in os.listdir(args.inversion_dir):
            inv_path = os.path.join(args.inversion_dir, inv_dir)
            if not os.path.isdir(inv_path):
                continue
            token_name = f"<{inv_dir}>"
            # check if the safetensors files are present
            assert "learned_embeds.safetensors" in os.listdir(inv_path), f"[image generator] learned_embeds.safetensors not found in {inv_path}."
            assert "learned_embeds_2.safetensors" in os.listdir(inv_path), f"[image generator] learned_embeds_2.safetensors not found in {inv_path}."
            # load inversion for both text encoders
            diffusion_pipeline.load_textual_inversion(os.path.join(inv_path, "learned_embeds.safetensors"), token=token_name, text_encoder=diffusion_pipeline.text_encoder, tokenizer=diffusion_pipeline.tokenizer)
            diffusion_pipeline.load_textual_inversion(os.path.join(inv_path, "learned_embeds_2.safetensors"), token=token_name, text_encoder=diffusion_pipeline.text_encoder_2, tokenizer=diffusion_pipeline.tokenizer_2)
        print("[inference] Pipeline loaded successfully.")
        print(f"Allocated Memory: {torch.cuda.memory_allocated() / 1e6} MB")
    else:
        print("[inference] Loading Stable Diffusion 2 pipeline...")
        diffusion_scheduler = EulerDiscreteScheduler.from_pretrained(args.sd_model, subfolder="scheduler")
        diffusion_pipeline = StableDiffusionPipeline.from_pretrained(args.sd_model, scheduler=diffusion_scheduler, torch_dtype=args.dtype)
        # load textual inversions
        inversion_dir = args.inversion_dir
        for inv_dir in os.listdir(inversion_dir):
            inv_path = os.path.join(inversion_dir, inv_dir)
            if not os.path.isdir(inv_path):
                continue
            token_name = f"<{inv_dir}>"
            diffusion_pipeline.load_textual_inversion(inv_path, token=token_name)
        print("[inference] Pipeline loaded successfully.")
        print(f"Allocated Memory: {torch.cuda.memory_allocated() / 1e6} MB")
    # configure pipeline
    diffusion_pipeline = diffusion_pipeline.to(device)
    if args.attn_slicing:
        diffusion_pipeline.enable_attention_slicing()
    if args.xformer:
        diffusion_pipeline.enable_xformers_memory_efficient_attention()
    diffusion_pipeline.enable_model_cpu_offload()
    # generate Initial Images
    init_images = generate_stable_diffusion(
        initial_prompts, pipeline=diffusion_pipeline, image_per_prompt=args.image_per_prompt,
        batch_size=args.batch_size, steps=args.init_steps, latents=init_latents)
    # release memory
    print(f"Allocated Memory: {torch.cuda.memory_allocated() / 1e6} MB")
    del diffusion_pipeline
    torch.cuda.empty_cache()
    print("[inference] Initial images generated successfully. Memory released.")
    print(f"Allocated Memory: {torch.cuda.memory_allocated() / 1e6} MB")
    # save initial images
    if args.save_process:
        os.makedirs(os.path.join(args.output_dir, "initial_images"), exist_ok=True)
        for i, image_batch in enumerate(init_images):
            os.makedirs(os.path.join(args.output_dir, "initial_images", f"{i}"), exist_ok=True)
            for j, image in enumerate(image_batch):
                image.save(os.path.join(args.output_dir, "initial_images", f"{i}", f"{j}.png"))
    exit()
    ## Mask Generation
    # load owlv2 model
    print("[inference] Loading object detection model...")
    detection_model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
    detection_processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    print("[inference] Object detection model loaded successfully.")
    print(f"Allocated Memory: {torch.cuda.memory_allocated() / 1e6} MB")
    # loop through the prompts
    mask_batches = []
    for id, image_batch, object_token in zip(prompt_ids, init_images, object_tokens):
        # upack object tokens
        classes = object_token["id_tokens"]
        # detect bounding boxes
        class_batch = [classes]*len(image_batch)
        bounding_batch = zero_shot_detection(
            image_batch, class_batch, processor=detection_processor, model=detection_model,
            threshold=0.2, device=device)
        # [showcase]: visualize results
        if args.show_process:
            visualize_boundings(image_batch, bounding_batch)
        # get the best bounding boxes for each class
        mask_bounding_batch = []
        for obj_classes, result, image in zip(class_batch, bounding_batch, image_batch):
            bounding_boxes = []
            for obj_class in obj_classes:
                if result[obj_class]: # check if the list is not empty
                    bounding_boxes.append(result[obj_class].pop(0))
                else:
                    bounding_boxes.append(None)
            mask_bounding_batch.append(bounding_boxes)
        # generate masks
        mask_batch = generate_masks(args.default_size, mask_bounding_batch)
        mask_batches.append(mask_batch)
        # save masks as images
        if args.save_process:
            os.makedirs(os.path.join(args.output_dir, "masks"), exist_ok=True)
            visualize_masks(mask_batch, image_batch, class_batch, save_path=os.path.join(args.output_dir, "masks", f"{id}.png"))
        # [showcase]: visualize masks
        if args.show_process:
            visualize_masks(mask_batch, image_batch, class_batch)
    print(f"Allocated Memory: {torch.cuda.memory_allocated() / 1e6} MB")
    # release memory
    del detection_model, detection_processor
    torch.cuda.empty_cache()
    
    print("[inference] Masks generated successfully. Memory released.")
    print(f"Allocated Memory: {torch.cuda.memory_allocated() / 1e6} MB")
    ## Impainting Concepts
    # load inpainting model
    if args.model_type == "sdxl":
        print("[inference] Loading Stable Diffusion XL inpainting pipeline...")
        inpaint_pipeline = AutoPipelineForInpainting.from_pretrained(args.inpaint_model, torch_dtype=torch.float16, variant="fp16")
        print("[inference] Loading Sdxl Textual Inversion.")
        # load textural inversions
        for inv_dir in os.listdir(args.inversion_dir):
            inv_path = os.path.join(args.inversion_dir, inv_dir)
            if not os.path.isdir(inv_path):
                continue
            token_name = f"<{inv_dir}>"
            # check if the safetensors files are present
            assert "learned_embeds.safetensors" in os.listdir(inv_path), f"[image generator] learned_embeds.safetensors not found in {inv_path}."
            assert "learned_embeds_2.safetensors" in os.listdir(inv_path), f"[image generator] learned_embeds_2.safetensors not found in {inv_path}."
            # load inversion for both text encoders
            inpaint_pipeline.load_textual_inversion(os.path.join(inv_path, "learned_embeds.safetensors"), token=token_name, text_encoder=inpaint_pipeline.text_encoder, tokenizer=inpaint_pipeline.tokenizer)
            inpaint_pipeline.load_textual_inversion(os.path.join(inv_path, "learned_embeds_2.safetensors"), token=token_name, text_encoder=inpaint_pipeline.text_encoder_2, tokenizer=inpaint_pipeline.tokenizer_2)
        print("[inference] Pipeline loaded successfully.")
        print(f"Allocated Memory: {torch.cuda.memory_allocated() / 1e6} MB")
    else:
        print("[inference] Loading Stable Diffusion 2 inpainting pipeline...")
        inpaint_pipeline = StableDiffusionInpaintPipeline.from_pretrained(args.inpaint_model, torch_dtype=torch.float16)
        # load textual inversions
        for inv_dir in os.listdir(inversion_dir):
            inv_path = os.path.join(inversion_dir, inv_dir)
            if not os.path.isdir(inv_path):
                continue
            token_name = f"<{inv_dir}>"
            inpaint_pipeline.load_textual_inversion(inv_path, token=token_name)
        print("[inference] Pipeline loaded successfully.")
        print(f"Allocated Memory: {torch.cuda.memory_allocated() / 1e6} MB")
    # configure pipeline
    inpaint_pipeline = inpaint_pipeline.to(device)
    if args.attn_slicing:
        inpaint_pipeline.enable_attention_slicing()
    if args.xformer:
        inpaint_pipeline.enable_xformers_memory_efficient_attention()
    inpaint_pipeline.enable_model_cpu_offload()
    # impaint
    for id, image_batch, mask_batch, object_token, style_token in zip(prompt_ids, init_images, mask_batches, object_tokens, style_tokens):
        # extract special tokens
        special_tokens = object_token["special_tokens"]
        # impanting concepts
        style_prompt = f" in {style_token} style." if style_token is not None else ""
        prompts = [[f"A {token}"+style_prompt for token in special_tokens] for _ in range(len(image_batch))]
        print(f"[inference] Impainting for prompt {prompts[0]}...")
        final_images = impaint_concept(
            image_batch, prompts, mask_batch, 
            pipeline=inpaint_pipeline, size=(args.width, args.height),
            steps=args.inpaint_steps, strength=args.inpaint_strength, mask_padding=args.mask_padding)
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
    print(f"Allocated Memory: {torch.cuda.memory_allocated() / 1e6} MB")
    # release memory
    del inpaint_pipeline
    torch.cuda.empty_cache()
    print(f"Allocated Memory: {torch.cuda.memory_allocated() / 1e6} MB")
    print("[inference] Inference completed.")