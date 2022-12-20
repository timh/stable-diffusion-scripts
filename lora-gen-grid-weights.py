from diffusers import StableDiffusionPipeline
from lora_diffusion import monkeypatch_lora, monkeypatch_replace_lora, tune_lora_scale
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

BASE_MODEL = "/home/tim/models/stable-diffusion-v1-5+vae"
LORA_UNET_PATH = Path("/home/tim/models/alex22-f222v-lora-batch1@1e-4,5e-5/lora_weight.pt")

SIZE = 512
PROMPT = "portrait photo of alexhin"
OUTPUT_FILENAME = "output.png"
OUTPUT_TEMP = "output-temp.png"
BASE_SEED = 2
NUM_IMAGES = 1

MIN = 0.0
MAX = 1.0
DIFF = MAX - MIN
NUM_STEPS = 10

if __name__ == "__main__":
    unet_path = Path(sys.argv[1]) if len(sys.argv) > 1 else LORA_UNET_PATH
    text_path = unet_path.with_suffix(".text_encoder.pt")

    print(f"unet_path {unet_path}")
    print(f"text_path {text_path}")

    pipe = StableDiffusionPipeline.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, safety_checker=None).to("cuda")
    monkeypatch_lora(pipe.unet, torch.load(unet_path))
    monkeypatch_lora(pipe.text_encoder, torch.load(text_path), target_replace_module=["CLIPAttention"])

    width = SIZE * NUM_STEPS
    height = SIZE * NUM_STEPS * NUM_IMAGES
    overall_image = Image.new(mode="RGB", size=[width, height])
    draw = ImageDraw.Draw(overall_image)
    font = ImageFont.truetype(Roboto, 30)

    time_start = datetime.datetime.now()
    for unet_idx in range(NUM_STEPS):
        unet_weight = unet_idx * (DIFF / (NUM_STEPS-1)) + MIN

        x = unet_idx * SIZE
        for text_idx in range(NUM_STEPS):
            text_weight = text_idx * (DIFF / (NUM_STEPS-1)) + MIN

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
        os.system(f"mv {OUTPUT_TEMP} {OUTPUT_FILENAME}")
    
    font = ImageFont.truetype(Roboto, 60)
    text = f"{unet_path.parent.name}/{unet_path.name}"
    draw.text(xy=(1, 451), text=text, font=font, fill="black")
    draw.text(xy=(0, 450), text=text, font=font, fill="white")
    overall_image.save(OUTPUT_TEMP)
    os.system(f"mv {OUTPUT_TEMP} {OUTPUT_FILENAME}")

    time_end = datetime.datetime.now()
    print(f"time taken: {time_end - time_start}")


