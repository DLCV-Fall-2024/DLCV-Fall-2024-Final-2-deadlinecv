import os, argparse, json
from typing import List, Tuple
from collections import deque
import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, StableDiffusionInpaintPipeline
from transformers import Owlv2ForObjectDetection, Owlv2Processor
from utils.image_generator import generate_stable_diffusion
from utils.mask_generator import zero_shot_detection, generate_masks, visualize_boundings, visualize_masks
from utils.concept_impainter import impaint_concept
import numpy as np
import matplotlib.pyplot as plt

def get_initial_tokens(prompt_token:str, prompt_eval:str, special_tokens:List[str]):
    '''
    Get the initial tokens which replace the special tokens by comparing the prompt with and without special tokens.
    Args:
        prompt_token (str): Prompt with special tokens.
        prompt_eval (str): Prompt without special tokens.
        special_tokens (List[str]): List of special tokens.
    Returns:
        List[str]: List of initial tokens corresponding to the special tokens.
    '''
    # use queues to store the words from the prompts
    prompt_token_words = deque(prompt_token.split())
    prompt_eval_words = deque(prompt_eval.split())
    initial_tokens = special_tokens.copy()
    # iterate through the words to find the initial tokens
    target_token = None
    init_token = ""
    while prompt_token_words and prompt_eval_words:
        if prompt_token_words[0] == prompt_eval_words[0]: # words match, pass
            if target_token is not None: # save the initial token if it exists
                target_index = special_tokens.index(target_token)
                initial_tokens[target_index] = init_token
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
        target_index = special_tokens.index(target_token)
        initial_tokens[target_index] = init_token

    return initial_tokens

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
        prompts_token = [data[prompt_id]["prompt"].rstrip('.').replace(">,", ">") for prompt_id in prompt_ids]
        prompts_eval = [data[prompt_id]["prompt_4_clip_eval"].rstrip('.') for prompt_id in prompt_ids]
        special_tokens = [data[prompt_id]["token_name"] for prompt_id in prompt_ids]
        initial_tokens = [get_initial_tokens(prompts_token[i], prompts_eval[i], special_tokens[i]) for i in range(len(prompt_ids))]
        # create custom prompts
        custom_prompts = [{
            "id": prompt_ids[i],
            "prompt": prompts_token[i],
            "prompt_eval": prompts_eval[i],
            "special_tokens": special_tokens[i],
            "initial_tokens": initial_tokens[i]
        } for i in range(len(prompt_ids))]
    return custom_prompts

def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for object detection.")
    parser.add_argument("--prompts", nargs="+", type=str, default=["A dog and a cat."], help="List of prompts for image generation.")
    parser.add_argument("--special_tokens", nargs="+", type=str, default=[], help="List of special tokens for textual inversion.")
    parser.add_argument("--initial_tokens", nargs="+", type=str, default=[], help="List of initial tokens for textual inversion.")
    parser.add_argument("--image_per_prompt", type=int, default=2, help="Number of images to generate per prompt.")
    parser.add_argument("--json", type=str, default=None, help="Path to the JSON file containing prompts. This will override the prompts argument.")
    parser.add_argument("--inversion_paths", nargs="+", type=str, default=[], help="List of paths to textual inversion embeddings.")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Path to the output directory.")
    args = parser.parse_args()
    # assertions
    assert args.image_per_prompt > 0, "[inference] Number of images per prompt should be greater than 0."
    assert len(args.special_tokens) == len(args.initial_tokens) == len(args.inversion_paths), "[inference] Number of special tokens, initial tokens, and inversion paths should be the same."
    # read JSON file
    if args.json is not None:
        custom_prompts = read_prompts_json(args.json)
    return args

if __name__ == "__main__":
    ## Initialization
    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[inference] Device: {device}.")
    torch.manual_seed(4219889)
    # directory
    code_dir = os.path.dirname(os.path.abspath(__file__))
    # parse arguments
    args = parse_args()
    # TODO: actually implement these in parse_args
    initial_prompt = ["A dog and a cat."]
    special_tokens = [["<dog>", "<cat2>"]]
    initial_tokens = [["dog", "cat"]]
    image_per_prompt = 2

    ## Generate Initial Images
    # TODO: handle style textual inversion
    # model
    print("[inference] Loading Stable Diffusion pipeline...")
    diffusion_scheduler = EulerDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="scheduler")
    diffusion_pipeline = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base", scheduler=diffusion_scheduler, torch_dtype=torch.float32)
    print("[inference] Pipeline loaded successfully. Generating initial images...")
    # generate images
    init_images = generate_stable_diffusion(initial_prompt, pipeline=diffusion_pipeline, image_per_prompt=image_per_prompt,
                                            batch_size=2, steps=25, attention_slicing=False, dtype=torch.float32)
    # clean up memory
    del diffusion_pipeline, diffusion_scheduler
    torch.cuda.empty_cache()
    print("[inference] Initial images generated successfully, memory released.")

    ## Post-Processing
    # models
    print("[inference] Loading object detection model...")
    detection_model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
    detection_processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    print("[inference] Object detection model loaded successfully.")
    print("[inference] Loading Stable Diffusion inpainting pipeline...")
    impaint_pipeline = StableDiffusionInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-2-inpainting", torch_dtype=torch.float32)
    print("[inference] Pipeline loaded successfully.")
    # load textual inversions
    textual_inversion_dir = os.path.join(code_dir, "textual_inversions", "sd2")
    inversion_lists = ["dog", "cat2"]
    inversion_paths = [os.path.join(textual_inversion_dir, inv) for inv in inversion_lists]
    inversion_tokens = ["<dog>", "<cat2>"]
    for inversion_path, inversion_token in zip(inversion_paths, inversion_tokens):
        impaint_pipeline.load_textual_inversion(inversion_path, token=inversion_token)
    
    for image_batch, classes, tokens in zip(init_images, initial_tokens, special_tokens):
        # detect bounding boxes
        class_batch = [classes]*image_per_prompt
        bounding_batch = zero_shot_detection(image_batch, class_batch, processor=detection_processor, model=detection_model, device=device)
        # [showcase]: visualize results
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
        mask_batch = generate_masks((512, 512), mask_bounding_batch)
        # [showcase]: visualize masks
        visualize_masks(mask_batch, image_batch, class_batch)
        # impanting concepts
        prompts = [[f"A {token}" for token in tokens] for _ in range(image_per_prompt)]
        final_images = impaint_concept(
            image_batch, prompts, mask_batch,
            pipeline=impaint_pipeline, steps=25,
            attention_slicing=False,
            device=device, dtype=torch.float32)

        # [showcase]: visualize final images
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
    # release memory
    del detection_model, detection_processor, impaint_pipeline
    torch.cuda.empty_cache()
    print("[inference] Memory released.")
    print("[inference] Inference completed.")