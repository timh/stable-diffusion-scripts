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
    CHECKPOINT_PATH=${OUTPUT_DIR}_`printf %05d $steps`.ckpt
    CHECKPOINT_PATH_ARCHIVE=${OUTPUT_DIR}/archive.ckpt
    if [ -f "$CHECKPOINT_PATH" ]
    then
      echo "$CHECKPOINT_PATH already exists, skipping..."
    else
      # create the CKPT, which is actually a zip file. some consumers of the CKPT file expect 
      # the top directory to be named "archive", so fake it with symlinks.
      echo "creating $CHECKPOINT_PATH..."
      ln -s "$MODEL_PATH" archive
      python ../../scripts/convert_diffusers_to_original_stable_diffusion.py --model_path archive --checkpoint_path $CHECKPOINT_PATH_ARCHIVE
      mv $CHECKPOINT_PATH_ARCHIVE $CHECKPOINT_PATH
      echo "$CHECKPOINT_PATH created"

      # add the text files with training parameters to the checkpoint
      zip -u $CHECKPOINT_PATH archive/*.txt
      rm -f archive
    fi
    ln -sf "$CHECKPOINT_PATH" "$TOP/stable-diffusion-webui/models/Stable-diffusion/"
  fi
)
done
