import argparse
import sys
import os
import numpy as np
sys.path.append('./')
import torch
from diffusers import StableDiffusionPipeline, AutoencoderKL
from typing import List
from PIL import Image, ImageFont, ImageDraw
from pathlib import Path
from src import diffuser_training 
import math

from fonts.ttf import Roboto

DEFAULT_SEED = 1
INF_STEPS = 50
CFG = 6.0
FONT_SIZE = 30
FONT_SIZE_BIG = 50
def sample(ckpt, delta_ckpts, prompt, token, freeze_model, seed):
    model_id = ckpt

    big_image: Image.Image = None

    num_width = min(4, len(delta_ckpts))
    num_height = math.ceil(len(delta_ckpts) / num_width)

    font_small = ImageFont.truetype(Roboto, FONT_SIZE)
    font_big = ImageFont.truetype(Roboto, FONT_SIZE_BIG)

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None).to("cuda")
    # pipe.vae = AutoencoderKL.from_pretrained("/home/tim/models/sd-vae-ft-mse", torch_dtype=torch.float16).to("cuda")
    # print(f"pipe.vae = {pipe.vae.__class__.__name__}")
    for delta_idx, delta_ckpt in enumerate(delta_ckpts):
        print(f"{delta_ckpt}:")
        diffuser_training.load_model(pipe.text_encoder, pipe.tokenizer, pipe.unet, delta_ckpt, token, freeze_model)

        generator = torch.Generator("cuda").manual_seed(seed)
        image: Image.Image = pipe(prompt, generator=generator, num_inference_steps=INF_STEPS, guidance_scale=CFG, eta=1., num_images_per_prompt=1).images[0]
        if big_image is None:
            width = image.width * num_width
            height = image.height * num_height
            big_image = Image.new(mode="RGB", size=(width, height))
            draw = ImageDraw.Draw(big_image)

        x = (delta_idx % num_width) * image.width
        y = int(delta_idx / num_width) * image.height
        big_image.paste(image, (x, y))

        filename_short = delta_ckpt.replace("/home/tim/models/", "").replace("/", "/\n ")
        text = filename_short
        textx = x + 2
        texty = y + 10
        draw.text(xy=(textx+1, texty+1), text=text, font=font_small, fill="black")
        draw.text(xy=(textx, texty), text=text, font=font_small, fill="white")

    text = f"'{prompt}'\ncfg: {CFG}, inference steps: {INF_STEPS}, seed: {seed}"
    textx = 2
    texty = height - FONT_SIZE_BIG * len(text.split("\n")) - 2
    draw.text(xy=(textx + 1, texty + 1), text=text, font=font_big, fill="black")
    draw.text(xy=(textx, texty), text=text, font=font_big, fill="white")

    big_image.save("output-temp.png")
    os.system("mv output-temp.png output.png")



def parse_args():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--ckpt', help='base model', type=str)
    parser.add_argument('--token', help='token, e.g. <new1>', type=str, default="<new1>")
    parser.add_argument('--delta_ckpt', dest='delta_ckpts', help='delta.bin', default=None, nargs='+')
    parser.add_argument('--prompt', help='prompt to generate', default=None, type=str)
    parser.add_argument('--freeze_model', help='crossattn or crossattn_kv', default='crossattn_kv', type=str)
    parser.add_argument('--seed', help='random seed', default=DEFAULT_SEED, type=int)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    sample(args.ckpt, args.delta_ckpts, args.prompt, args.token, args.freeze_model, args.seed)
