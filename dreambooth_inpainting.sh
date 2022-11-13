set -e

export STEPS=6000
export TOP="/workspace"

export MODEL_NAME="runwayml/stable-diffusion-inpainting"
export INSTANCE_DIR="$TOP/images.alex-20"
export OUTPUT_DIR="$TOP/outputs/awh20-inpainting"
export CLASS_DIR="$TOP/class_images/person_ddim"

echo "starting inpainting training: output $OUTPUT_DIR, $STEPS steps.."
accelerate launch train_inpainting_dreambooth.py \
  --output_dir=$OUTPUT_DIR \
  --class_data_dir=$CLASS_DIR \
  --instance_data_dir=$INSTANCE_DIR \
  --instance_prompt="a photo of awh" \
  --class_prompt="a photo of person" \
  --learning_rate=1e-6 \
  --save_interval=500 \
  --save_infer_steps=35 \
\
  --seed=1 \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --resolution=512 \
  --train_batch_size=1 \
  --train_text_encoder \
  --sample_batch_size=1 \
  --gradient_accumulation_steps=1 --gradient_checkpointing \
  --use_8bit_adam \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=$STEPS \
  --mixed_precision=fp16 \
  --not_cache_latents \
  --hflip

