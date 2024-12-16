export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export OUTPUT_DIR="./output/finetune/lora/image_pokemon_blip_captions"
# export TRAIN_DATA_DIR="./image/single_image/"
# export HUB_MODEL_ID="naruto-lora"
export DATASET_NAME="svjack/pokemon-blip-captions-en-zh"

accelerate launch --mixed_precision="no"  ./ldm_with_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --dataloader_num_workers=8 \
  --resolution=512 \
  --center_crop \
  --random_flip \
  --train_batch_size=8 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=15000 \
  --learning_rate=1e-03 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --report_to=wandb \
  --checkpointing_steps=5000 \
  --caption_column="eh_text" \
  --seed=1024