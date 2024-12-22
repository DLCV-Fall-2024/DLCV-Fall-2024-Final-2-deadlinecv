'''
This script downloads the sd2 models from the Hugging Face Hub.
You can use this to download the models to your local machine and pre-upload them to Colab or Kaggle for faster inference.
If you need sdxl or other models, you can change the directory name and repo ids for that.
Also, the ignore patterns list is used to exclude certain unecessary files from the download, change them as needed.
'''
import os
from huggingface_hub import snapshot_download

# ignore patterns
allow_patterns = ["*.fp16.safetensors", "*.txt", "*.json"]

# stable-diffusion-xl-base-1.0
sdxl_base_dir = "stable-diffusion-xl-base-1.0"
os.makedirs(sdxl_base_dir, exist_ok=True)
model_path = snapshot_download(repo_id="stabilityai/stable-diffusion-xl-base-1.0", local_dir=sdxl_base_dir, allow_patterns=allow_patterns)
print(f"Model downloaded to: {model_path}")
# stable-diffusion-2-inpainting
sdxl_inpainting_dir = "stable-diffusion-xl-1.0-inpainting-0.1"
os.makedirs(sdxl_inpainting_dir, exist_ok=True)
model_path = snapshot_download(repo_id="diffusers/stable-diffusion-xl-1.0-inpainting-0.1", local_dir=sdxl_inpainting_dir, allow_patterns=allow_patterns)
print(f"Model downloaded to: {model_path}")