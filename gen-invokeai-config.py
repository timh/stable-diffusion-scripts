import os
from typing import List

OUTPUT_DIR = "/workspace/outputs"
INVOKEAI_CONFIG_DIR = "/workspace/InvokeAI/configs"
INVOKEAI_CONFIG_FILENAME = INVOKEAI_CONFIG_DIR + "/models.yaml"

checkpoints: List[str] = [fn for fn in os.listdir(OUTPUT_DIR) if fn.endswith(".ckpt")]

with open(INVOKEAI_CONFIG_FILENAME, "w") as file:
    file.write("""
# stable-diffusion-1.4:
#   description: Stable Diffusion inference model version 1.4
#   config: configs/stable-diffusion/v1-inference.yaml
#   weights: models/ldm/stable-diffusion-v1/sd-v1-4.ckpt
#   vae: models/ldm/stable-diffusion-v1/vae-ft-mse-840000-ema-pruned.ckpt
#   width: 512
#   height: 512
stable-diffusion-1.5:
  weights: ./models/ldm/stable-diffusion-v1/v1-5-pruned-emaonly.ckpt
  config: ./configs/stable-diffusion/v1-inference.yaml
  width: 512
  height: 512
  vae: ./models/ldm/stable-diffusion-v1/vae-ft-mse-840000-ema-pruned.ckpt
  default: true
  description: The newest Stable Diffusion version 1.5 weight file (4.27 GB)
inpainting-1.5:
  weights: ./models/ldm/stable-diffusion-v1/sd-v1-5-inpainting.ckpt
  config: configs/stable-diffusion/v1-inpainting-inference.yaml
  vae: models/ldm/stable-diffusion-v1/vae-ft-mse-840000-ema-pruned.ckpt
  width: 512
  height: 512
  description: RunwayML SD 1.5 model optimized for inpainting

""")

    for filename in sorted(checkpoints):
        name = filename.replace(".ckpt", "")
        config = "configs/stable-diffusion/v1-inference.yaml"
        if "inpainting" in filename:
            config = "configs/stable-diffusion/v1-inpainting-inference.yaml"
        file.write(f"""
{name}:
  weights: {OUTPUT_DIR}/{filename}
  config: {config}
  vae: models/ldm/stable-diffusion-v1/vae-ft-mse-840000-ema-pruned.ckpt
  width: 512
  height: 512
""")
        
