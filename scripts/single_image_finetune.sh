export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export OUTPUT_DIR="./output/finetune/lora/single_image_test"
export TRAIN_DATA_DIR="./image/test/"
# export HUB_MODEL_ID="naruto-lora"
# export DATASET_NAME="lambdalabs/naruto-blip-captions"

accelerate launch --mixed_precision="no"  ./ldm_with_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DATA_DIR \
  --dataloader_num_workers=8 \
  --resolution=512 \
  --center_crop \
  --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=20 \
  --learning_rate=1e-03 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --report_to=wandb \
  --checkpointing_steps=2 \
  --validation_prompt="He is qyf" \
  --seed=1337