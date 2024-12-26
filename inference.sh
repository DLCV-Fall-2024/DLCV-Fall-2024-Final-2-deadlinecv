#!/bin/bash
python3 inference.py \
    --json $1 \
    --prompt_id "0" \
    --special_tokens "<cat2>" "<dog6>" \
    --init_tokens "cat" "cat" \
    --id_tokens "cat" "cat" \
    --inversion_dir textual_inversions/sdxl \
    --image_per_prompt 10 \
    --seeds 7 5 43 45 72 42 32 55 35 14 \
    --output_dir $2 \
    --inpaint_strength 0.8 \
    --init_steps 25 \
    --inpaint_steps 50 \
    --seed 36 \
    --batch_size 1 \
    --mask_padding 32
python3 inference.py \
    --json $1 \
    --prompt_id "1" \
    --special_tokens "<flower_1>" "<vase>" \
    --init_tokens "white flower" "tall red vase" \
    --id_tokens "flower" "vase" \
    --inversion_dir textual_inversions/sdxl \
    --image_per_prompt 10 \
    --seeds 12 0 41 9 108 116 85 1 99 39 \
    --output_dir $2 \
    --inpaint_strength 0.8 \
    --init_steps 25 \
    --inpaint_steps 50 \
    --seed 36 \
    --batch_size 1 \
    --mask_padding 32
python3 inference.py \
    --json $1 \
    --prompt_id "2" \
    --special_tokens "<dog>" "<pet_cat1>" "<dog6>" \
    --init_tokens "little dog" "cat" "little dog" \
    --id_tokens "dog" "cat" "dog" \
    --inversion_dir textual_inversions/sdxl \
    --image_per_prompt 10 \
    --seeds 109 101 116 113 137 134 148 149 141 158 \
    --output_dir $2 \
    --inpaint_strength 0.8 \
    --init_steps 50 \
    --inpaint_steps 50 \
    --seed 1126 \
    --batch_size 1 \
    --mask_padding 32
python3 inference.py \
    --json $1 \
    --prompt_id "3" \
    --special_tokens "<cat2>" "<wearable_glasses>" \
    --init_tokens "grey cat" "brown glasses" \
    --id_tokens "cat" "glasses" \
    --inversion_dir textual_inversions/sdxl \
    --image_per_prompt 10 \
    --seeds 641 533 521 585 538 555 667 584 519 610 \
    --output_dir $2 \
    --inpaint_strength 0.6 \
    --init_steps 25 \
    --inpaint_steps 50 \
    --seed 4129889 \
    --batch_size 1