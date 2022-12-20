import argparse
import sys
import os, os.path
import numpy as np
sys.path.append('./')
import torch
from diffusers import StableDiffusionPipeline, AutoencoderKL, DPMSolverMultistepScheduler
from typing import List
from PIL import Image, ImageFont, ImageDraw
from pathlib import Path
from src import diffuser_training 
import math

from fonts.ttf import Roboto

DEFAULT_SEED = 1
INF_STEPS = 30
CFG = 6.0
FONT_SIZE = 24
FONT_SIZE_BIG = 40

def sample(model_id: str, delta_ckpts: List[str], prompt: str, token: str, freeze_model: str, seed: int, output: str):
    num_width = min(4, len(delta_ckpts))
    num_height = math.ceil(len(delta_ckpts) / num_width)

    font_small = ImageFont.truetype(Roboto, FONT_SIZE)
    font_big = ImageFont.truetype(Roboto, FONT_SIZE_BIG)
    out_image: Image.Image = None

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type='dpmsolver++', solver_order=1)

    common_root = os.path.commonprefix([Path(d).parent for d in delta_ckpts])

    for delta_idx, delta_ckpt in enumerate(delta_ckpts):
        print(f"{delta_ckpt}:")
        diffuser_training.load_model(pipe.text_encoder, pipe.tokenizer, pipe.unet, delta_ckpt, token, freeze_model)

        generator = torch.Generator("cuda").manual_seed(seed)
        image: Image.Image = pipe(prompt, generator=generator, num_inference_steps=INF_STEPS, guidance_scale=CFG, eta=1., num_images_per_prompt=1).images[0]
        if out_image is None:
            width = image.width * num_width
            height = image.height * num_height
            out_image = Image.new(mode="RGB", size=(width, height))
            draw = ImageDraw.Draw(out_image)

        x = (delta_idx % num_width) * image.width
        y = int(delta_idx / num_width) * image.height
        out_image.paste(image, (x, y))

        text = delta_ckpt.replace("/home/tim/models/", "").replace("/", "/\n ")
        textx = x + 2
        texty = y + 10
        draw.text(xy=(textx+1, texty+1), text=text, font=font_small, fill="black")
        draw.text(xy=(textx, texty), text=text, font=font_small, fill="white")

    text = f"'{prompt}'\n"
    text += f"cfg: {CFG}, inference steps: {INF_STEPS}, seed: {seed}\n"
    text += f"base: {Path(model_id).name}"

    textx = width - 2
    texty = height - 2
    draw.text(xy=(textx + 1, texty + 1), text=text, font=font_big, fill="black", anchor="rd")
    draw.text(xy=(textx, texty), text=text, font=font_big, fill="white", anchor="rd")

    output_temp = f"{output}-temp.png"
    out_image.save(output_temp)
    os.system(f"mv {output_temp} {output}")

    print(f"created {output}")



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model_id", help="base diffusers model name or path", default="runwayml/stable-diffusion-v1-5", type=str)
    parser.add_argument("--token", help="token, e.g. <new1>", type=str)
    parser.add_argument("--delta", dest="delta_ckpts", help="filename(s) to delta.bin", default=None, nargs='+')
    parser.add_argument("--prompt", help="prompt to generate", default=None, type=str)
    parser.add_argument("--freeze_model", help="crossattn or crossattn_kv", default="crossattn_kv", type=str)
    parser.add_argument("--seed", help="random seed", default=DEFAULT_SEED, type=int)
    parser.add_argument("-o", "--out", "--output", dest="output", help="filename of output PNG", default="output.png", type=str)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    sample(model_id=args.model_id, delta_ckpts=args.delta_ckpts, prompt=args.prompt, token=args.token, freeze_model=args.freeze_model, seed=args.seed, output=args.output)
