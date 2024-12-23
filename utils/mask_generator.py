import os
from PIL import Image
from typing import List, Tuple
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import matplotlib.pyplot as plt
import numpy as np

def zero_shot_detection(images:List[Image.Image], classes:List[List[str]],
                              processor:Owlv2Processor=None, model:Owlv2ForObjectDetection=None,
                              device:str="cuda", threshold:float=0.2) -> List[dict]:
    '''
    Perform zero-shot object detection using the OWL-V2 model.
    Args:
        images (List[Image.Image]): List of PIL images to perform object detection on.
        classes (List[List[str]]): List of classes to detect in each image.
        processor (Owlv2Processor): Processor for the OWL-V2 model. Default is "google/owlv2-base-patch16-ensemble".
        model (Owlv2ForObjectDetection): OWL-V2 model for object detection. Default is "google/owlv2-base-patch16-ensemble".
        device (str): Device to run the model on. Default is "cuda".
        threshold (float): Confidence threshold for object detection. Default is 0.2. If you want to handle it yourself, set it to 0.
    Returns:
        List[dict]: List of dictionaries containing the detected objects and their bounding boxes.
    '''
    # check inputs
    assert len(images) > 0, "[mask generator] No images provided."
    assert len(classes) > 0, "[mask generator] No classes provided."
    assert len(images) == len(classes), "[mask generator] Number of images and classes list should be the same."
    assert threshold >= 0 and threshold <= 1, "[mask generator] Threshold should be between 0 and 1."
    # prepare model and processor
    if model is None or processor is None:
        print("[mask generator] No model or processor provided. Loading default model and processor...")
        model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
        processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        print("[mask generator] Model and processor loaded successfully.")
    model.to(device)
    model.eval()
    # preprocess images
    inputs = processor(text=classes, images=images, return_tensors="pt").to(device)
    # zero-shot object detection
    with torch.no_grad():
        outputs = model(**inputs)
    # retrieve results
    logits = torch.max(outputs["logits"], dim=-1)
    scores = torch.sigmoid(logits.values).cpu().detach().numpy()
    labels = logits.indices.cpu().detach().numpy()
    boxes = outputs["pred_boxes"].cpu().detach().numpy()
    # post-process results
    results = []
    for i in range(len(images)):
        label_results = {label:[] for label in classes[i]}
        for j in range(len(scores[i])):
            if scores[i][j] >= threshold: # filter out low-confidence predictions
                label = classes[i][labels[i][j]]
                label_results[label].append({
                    "score": scores[i][j],
                    "box": boxes[i][j]
                })
        # sort results by score in descending order
        for label in label_results:
            label_results[label] = sorted(label_results[label], key=lambda x: x["score"], reverse=True)
        results.append(label_results)
    
    return results

def visualize_boundings(images:List[Image.Image], bounding_boxes:List[dict]):
    '''
    Visualize the bounding boxes on the images. This function is not necessary, just for verification and showcasing purposes.
    Args:
        images (List[Image.Image]): List of PIL images to visualize.
        bounding_boxes (List[dict]): List of dictionaries containing the bounding boxes for each image.
    '''
    assert len(images) == len(bounding_boxes), "[mask generator] Batch size of images and bounding boxes should be the same."
    # prepare figure
    num_rows = np.ceil(np.sqrt(len(images))).astype(int)
    num_columns = np.ceil(len(images) / num_rows).astype(int)
    _, axes = plt.subplots(num_rows, num_columns, squeeze=False)
    axes = axes.flatten()
    # plot images
    for i, (image, box) in enumerate(zip(images, bounding_boxes)):
        ax = axes[i]
        ax.imshow(image, extent=(0, 1, 1, 0))
        ax.axis("off")
        # extract results
        for label in box:
            for item in box[label]:
                cx, cy, w, h = item["box"]
                score = item["score"]
                # plot bounding box
                ax.plot([cx - w/2, cx + w/2, cx + w/2, cx - w/2, cx - w/2],
                        [cy - h/2, cy - h/2, cy + h/2, cy + h/2, cy - h/2],
                        color="red", linewidth=2)
                # add label
                ax.text(cx - w/2, cy - h/2 + 0.015,
                        f"{label} ({score:.2f})", color="red",
                        ha="left", va="top",
                        bbox={
                            "facecolor": "white",
                            "edgecolor": "red",
                            "boxstyle": "square,pad=.3"
                        })
    # turn off unused axes
    for ax in axes[len(images):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def generate_masks(image_size:Tuple[int, int], bounding_batch:List[List[dict]]):
    '''
    Generate masks from the bounding boxes.
    Args:
        image_size (Tuple[int, int]): Size of the image to generate masks for.
        bounding_batch (List[List[dict]]): List of list of dictionaries containing the bounding boxes for each image.
    Returns:
        List[List[np.ndarray]]: List of list of masks for each image.
    '''
    # generate masks
    mask_batch = []
    for boundings in bounding_batch:
        masks = []
        for bounding in boundings:
            if bounding is not None:
                mask = np.zeros(image_size, dtype=np.uint8)
                cx, cy, w, h = bounding["box"]
                xmin = np.clip(int((cx - w/2) * image_size[0]), 0, image_size[0])
                xmax = np.clip(int((cx + w/2) * image_size[0]), 0, image_size[0])
                ymin = np.clip(int((cy - h/2) * image_size[1]), 0, image_size[1])
                ymax = np.clip(int((cy + h/2) * image_size[1]), 0, image_size[1])
                mask[ymin:ymax, xmin:xmax] = 255
                masks.append(mask)
            else:
                masks.append(None)
        mask_batch.append(masks)
    return mask_batch

def visualize_masks(mask_batch:List[List[np.ndarray]], image_batch:List[Image.Image], class_batch:List[List[str]], save_path:str=None):
    '''
    Visualize the masks on the images. This function is not necessary, just for verification and showcasing purposes.
    Args:
        mask_batch (List[List[np.ndarray]]): List of list of masks to visualize.
        image_batch (List[Image.Image]): List of PIL images to visualize.
        class_batch (List[List[str]]): List of classes for each image.
        save_path (str): Path to save the visualization. Default is None.
    '''
    assert len(mask_batch) == len(image_batch), "[mask generator] Batch size of masks and images should be the same."
    # prepare figure
    num_rows = np.ceil(np.sqrt(len(image_batch))).astype(int)
    num_columns = np.ceil(len(image_batch) / num_rows).astype(int)
    _, axes = plt.subplots(num_rows, num_columns, squeeze=False)
    axes = axes.flatten()
    # plot images
    for i, (masks, image, classes) in enumerate(zip(mask_batch, image_batch, class_batch)):
        ax = axes[i]
        ax.imshow(image, alpha=0.8)
        ax.axis("off")
        # draw masks in different colors
        handles = []
        for j, (mask, label) in enumerate(zip(masks, classes)):
            if mask is not None:
                color = np.array(plt.cm.tab10(j % 10))
                colored_mask = np.dstack([color[i]*mask/255 for i in range(4)])
                ax.imshow(colored_mask, alpha=0.5)
                handles.append(plt.Line2D([0], [0], color=color, lw=4, label=label))
        # add legends
        ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=True)

    # turn off unused axes
    for ax in axes[len(image_batch):]:
        ax.axis("off")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

# example usage
if __name__ == "__main__":
    # paths
    code_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(code_dir, os.pardir, "Data", "concept_image", "vase")
    # load images
    image_batch = [Image.open(os.path.join(images_dir, filename)).convert("RGB").resize((512, 512)) 
              for filename in ["00.jpg", "01.jpg", "02.jpg"]]
    class_batch = [["vase", "pot", "flower", "vase"]] * len(image_batch)
    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}.")
    # get bounding boxes via owlv2 zero-shot object detection
    bounding_batch = zero_shot_detection(image_batch, class_batch, device=device)
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



