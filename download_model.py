'''
This script downloads the sd2 models from the Hugging Face Hub.
You can use this to download the models to your local machine and pre-upload them to Colab or Kaggle for faster inference.
If you need sdxl or other models, you can change the directory name and repo ids for that.
Also, the ignore patterns list is used to exclude certain unecessary files from the download, change them as needed.
'''
import os
from huggingface_hub import snapshot_download

# ignore patterns
ignore_patterns = ["*fp16*", "*.bin", "*.ckpt", "*pruned*", "*ema*"]
# stable-diffusion-2-1-base
sd2_1_base_dir = "stable-diffusion-2-1-base"
os.makedirs(sd2_1_base_dir, exist_ok=True)
model_path = snapshot_download(repo_id="stabilityai/stable-diffusion-2-1-base", local_dir=sd2_1_base_dir, ignore_patterns=ignore_patterns)
print(f"Model downloaded to: {model_path}")
# stable-diffusion-2-inpainting
sd2_inpainting_dir = "stable-diffusion-2-inpainting"
os.makedirs(sd2_inpainting_dir, exist_ok=True)
model_path = snapshot_download(repo_id="stabilityai/stable-diffusion-2-inpainting", local_dir=sd2_inpainting_dir, ignore_patterns=ignore_patterns)
print(f"Model downloaded to: {model_path}")