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

# BASE_MODEL = "/home/tim/models/stable-diffusion-v1-5+vae"
BASE_MODEL = "/home/tim/models/ppp"

SIZE = 512
# PROMPT = "portrait photo of alexhin"
PROMPT = "portrait photo of <alex>"
OUTPUT_TEMP = "outputs/output-temp.png"
BASE_SEED = 0
NUM_IMAGES = 1

MIN_UNET = 0.6
MAX_UNET = 1.0
DIFF_UNET = MAX_UNET - MIN_UNET

MIN_TEXT = 0.0
MAX_TEXT = 1.0
DIFF_TEXT = MAX_TEXT - MIN_TEXT

NUM_STEPS = 11

if __name__ == "__main__":
    # unet_path = Path(sys.argv[1]) if len(sys.argv) > 1 else LORA_UNET_PATH
    # text_path = unet_path.with_suffix(".text_encoder.pt")
    unet_path = sys.argv[1]
    output = "outputs/" + Path(unet_path).parent.name + "--" + Path(unet_path).stem + ".png"

    print(f"unet_path {unet_path}")
    print(f"output {output}")

    pipe = StableDiffusionPipeline.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, safety_checker=None).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type='dpmsolver++', solver_order=1)
    patch_pipe(pipe, unet_path, "<alex>", patch_text=True, patch_ti=True, patch_unet=True)
    # monkeypatch_lora(pipe.unet, torch.load(unet_path))
    # monkeypatch_lora(pipe.text_encoder, torch.load(text_path), target_replace_module=["CLIPAttention"])

    width = SIZE * NUM_STEPS
    height = SIZE * NUM_STEPS * NUM_IMAGES
    overall_image = Image.new(mode="RGB", size=[width, height])
    draw = ImageDraw.Draw(overall_image)
    font = ImageFont.truetype(Roboto, 30)

    time_start = datetime.datetime.now()
    for unet_idx in range(NUM_STEPS):
        unet_weight = unet_idx * (DIFF_UNET / (NUM_STEPS-1)) + MIN_UNET

        x = unet_idx * SIZE
        for text_idx in range(NUM_STEPS):
            text_weight = text_idx * (DIFF_TEXT / (NUM_STEPS-1)) + MIN_TEXT

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
