set -e

if [ "$TOP"x = "x" -o "$OUTPUT_DIR"x = "x" ]
then
  echo "must define TOP and OUTPUT_DIR"
  exit 1
fi

for steps in `ls $OUTPUT_DIR`
do(
  if [ "$steps" = "0" ]
  then
    true
  else
    MODEL_PATH=$OUTPUT_DIR/$steps
    CHECKPOINT_PATH=${OUTPUT_DIR}_${steps}.ckpt
    if [ -f "$CHECKPOINT_PATH" ]
    then
      echo "$CHECKPOINT_PATH already exists, skipping..."
    else
      python ../../scripts/convert_diffusers_to_original_stable_diffusion.py --model_path $MODEL_PATH --checkpoint_path $CHECKPOINT_PATH
      echo "$CHECKPOINT_PATH created"
    fi
    ln -sf "$CHECKPOINT_PATH" "$TOP/stable-diffusion-webui/models/Stable-diffusion/"
  fi
)
done
