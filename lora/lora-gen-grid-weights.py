from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from lora_diffusion import monkeypatch_lora, monkeypatch_replace_lora, tune_lora_scale, patch_pipe
from PIL import Image, ImageFont, ImageDraw
from fonts.ttf import Roboto

from collections import namedtuple, deque
from typing import List, Deque, Dict
import torch
import textwrap
import os, sys, re
import math
from pathlib import Path
import datetime
import numpy as np

# BASE_MODEL = "/home/tim/models/stable-diffusion-v1-5+vae"
BASE_MODEL = "/home/tim/models/f222v"
# BASE_MODEL = "/home/tim/models/f222"

SIZE = 512
PROMPT = "portrait photo of <alex>"
OUTPUT_TEMP = "outputs/output-temp.png"
BASE_SEED = 0
NUM_IMAGES = 1

NUM_STEPS = 6
RANGE_UNET = list(np.linspace(0.5, 1.0, NUM_STEPS))
RANGE_TEXT = list(np.linspace(0.5, 1.0, NUM_STEPS))


if __name__ == "__main__":
    # unet_path = Path(sys.argv[1]) if len(sys.argv) > 1 else LORA_UNET_PATH
    # text_path = unet_path.with_suffix(".text_encoder.pt")
    unet_path = sys.argv[1]
    output = "outputs/" + Path(unet_path).absolute().parent.name + "--" + Path(unet_path).absolute().stem + ".png"
    Path("outputs").mkdir(exist_ok=True)
    print(f"unet_path {unet_path}")
    print(f"output {output}")

    pipe = StableDiffusionPipeline.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, safety_checker=None).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type='dpmsolver++', solver_order=1)
    patch_pipe(pipe, unet_path, "<alex>", patch_text=True, patch_ti=True, patch_unet=True)

    width = SIZE * NUM_STEPS
    height = SIZE * NUM_STEPS * NUM_IMAGES
    overall_image = Image.new(mode="RGB", size=[width, height])
    draw = ImageDraw.Draw(overall_image)
    font = ImageFont.truetype(Roboto, 30)

    time_start = datetime.datetime.now()
    for unet_idx, unet_weight in enumerate(RANGE_UNET):
        x = unet_idx * SIZE
        for text_idx, text_weight in enumerate(RANGE_TEXT):
            tune_lora_scale(pipe.unet, unet_weight)
            tune_lora_scale(pipe.text_encoder, text_weight)

            print(f"gen @ unet {unet_weight:.2}, text_encoder {text_weight:.2}")

            generator = torch.Generator("cuda").manual_seed(BASE_SEED)
            images = pipe(PROMPT, generator=generator, num_inference_steps=30, guidance_scale=7, num_images_per_prompt=NUM_IMAGES).images

            for image_idx, image in enumerate(images):
                y = (image_idx * NUM_STEPS * SIZE) + (text_idx * SIZE)
                overall_image.paste(image, (x, y))
                text = f"unet {unet_weight:.2}\ntext {text_weight:.2}"
                draw.text(xy=(x + 1, y + 1), text=text, font=font, fill="black")
                draw.text(xy=(x, y), text=text, font=font, fill="white")

        overall_image.save(OUTPUT_TEMP)
        os.system(f"mv {OUTPUT_TEMP} {output}")
    
    font = ImageFont.truetype(Roboto, 60)
    text = f"{output}"
    draw.text(xy=(1, 451), text=text, font=font, fill="black")
    draw.text(xy=(0, 450), text=text, font=font, fill="white")
    overall_image.save(OUTPUT_TEMP)
    os.system(f"mv {OUTPUT_TEMP} {output}")

    time_end = datetime.datetime.now()
    print(f"time taken: {time_end - time_start}")
