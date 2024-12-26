git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
cd examples
cd textual_inversion
pip install -r requirements.txt
accelerate config

accelerate launch textual_inversion_sdxl.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --train_data_dir="../../../Data/concept_image/cat2" \
  --learnable_property="object" \
  --placeholder_token="<cat2>" \
  --initializer_token="cat" \
  --mixed_precision="bf16" \
  --resolution=768 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=1000 \
  --learning_rate=5.0e-04 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --save_as_full_pipeline \
  --output_dir="./textual_inversion_cat2_sdxl"

accelerate launch textual_inversion_sdxl.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --train_data_dir="../../../Data/concept_image/dog" \
  --learnable_property="object" \
  --placeholder_token="<dog>" \
  --initializer_token="dog" \
  --mixed_precision="bf16" \
  --resolution=768 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=1000 \
  --learning_rate=5.0e-04 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --save_as_full_pipeline \
  --output_dir="./textual_inversion_dog_sdxl"

accelerate launch textual_inversion_sdxl.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --train_data_dir="../../../Data/concept_image/dog6" \
  --learnable_property="object" \
  --placeholder_token="<dog6>" \
  --initializer_token="dog" \
  --mixed_precision="bf16" \
  --resolution=768 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=1000 \
  --learning_rate=5.0e-04 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --save_as_full_pipeline \
  --output_dir="./textual_inversion_dog6_sdxl"

accelerate launch textual_inversion_sdxl.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --train_data_dir="../../../Data/concept_image/flower_1" \
  --learnable_property="object" \
  --placeholder_token="<flower_1>" \
  --initializer_token="flower" \
  --mixed_precision="bf16" \
  --resolution=768 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=1000 \
  --learning_rate=5.0e-04 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --save_as_full_pipeline \
  --output_dir="./textual_inversion_flower_1_sdxl"

accelerate launch textual_inversion_sdxl.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --train_data_dir="../../../Data/concept_image/pet_cat1" \
  --learnable_property="object" \
  --placeholder_token="<pet_cat1>" \
  --initializer_token="cat" \
  --mixed_precision="bf16" \
  --resolution=768 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=1000 \
  --learning_rate=5.0e-04 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --save_as_full_pipeline \
  --output_dir="./textual_inversion_pet_cat1_sdxl"

accelerate launch textual_inversion_sdxl.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --train_data_dir="../../../Data/concept_image/vase" \
  --learnable_property="object" \
  --placeholder_token="<vase>" \
  --initializer_token="vase" \
  --mixed_precision="bf16" \
  --resolution=768 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=1000 \
  --learning_rate=5.0e-04 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --save_as_full_pipeline \
  --output_dir="./textual_inversion_vase_sdxl"

accelerate launch textual_inversion_sdxl.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --train_data_dir="../../../Data/concept_image/watercolor" \
  --learnable_property="style" \
  --placeholder_token="<watercolor>" \
  --initializer_token="watercolor" \
  --mixed_precision="bf16" \
  --resolution=768 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=1000 \
  --learning_rate=5.0e-04 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --save_as_full_pipeline \
  --output_dir="./textual_inversion_watercolor_sdxl"

accelerate launch textual_inversion_sdxl.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --train_data_dir="../../../Data/concept_image/wearable_glasses" \
  --learnable_property="object" \
  --placeholder_token="<wearable_glasses>" \
  --initializer_token="glasses" \
  --mixed_precision="bf16" \
  --resolution=768 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=1000 \
  --learning_rate=5.0e-04 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --save_as_full_pipeline \
  --output_dir="./textual_inversion_glasses_sdxl"

cd ../../../
  