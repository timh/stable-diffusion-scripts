import sys
import os
import os.path
import re
import subprocess
import argparse
import shlex
import time
import datetime
import json
from collections import namedtuple, deque
from typing import List

import txt2img
from txt2img import ImageSet

# ddim, k_dpm_2_a, k_dpm_2, k_euler_a, k_euler, k_heun, k_lms, plms

# generator for combinatorial set of images to generate
def gen_renders(config: argparse.Namespace):
    for model in config.models:
        for prompt_idx, prompt in enumerate(config.prompts):
            for sampler in config.samplers:
                for cfg in config.cfgs:
                    negative_prompt = None
                    if len(config.negative_prompts) == 1:
                        negative_prompt = config.negative_prompts[0]
                    elif len(config.negative_prompts) > 1:
                        negative_prompt = config.negative_prompts[prompt_idx]

                    one = ImageSet(root_output_dir=config.output_dir, 
                                    prompt=prompt, negative_prompt=negative_prompt,
                                    model_dir=model, 
                                    sampler_str=sampler, guidance_scale=cfg,
                                    seed=config.base_seed, num_images=config.num_images,
                                    width=config.width, height=config.height)
                    yield one

def gen(image_gen: txt2img.ImageGenerator, config: argparse.Namespace):
    for one in gen_renders(config):
        time_start = time.perf_counter()
        num_generated = image_gen.gen_images(one)
        time_end = time.perf_counter()

        if num_generated > 0:
            filename = f"{one.output_dir}/gen-many.json"
            if os.path.exists(filename):
                stats_root = json.load(open(filename, "r"))
            else:
                stats_root = {}

            if 'runs' not in stats_root:
                stats_root['runs'] = []
            
            stats_array = stats_root['runs']
            with open(filename, "w") as file:
                stats = {}
                stats['timestamp'] = datetime.datetime.now().ctime()
                stats['argv'] = sys.argv
                stats['config_args'] = config.config_args
                stats['num_generated'] = num_generated
                stats['image_set'] = {
                    'output_dir': one.output_dir,
                    'prompt': one.prompt,
                    'negative_prompt': one.negative_prompt,
                    'model_dir': one.model_dir,
                    'sampler_name': one.sampler_name,
                    'sampler_steps': one.sampler_steps,
                    'guidance_scale': one.guidance_scale,
                    'base_seed': config.base_seed,
                    'width': config.width,
                    'height': config.height,
                }
                stats['timing'] = {
                    'total': (time_end - time_start),
                    'per_image': (time_end - time_start) / num_generated,
                }
                stats_array.append(stats)
                json.dump(stats_root, file, indent=2)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="gen many sample images")
    parser.add_argument("-p", "--prompt", dest='prompts', nargs='+', action='append', required=True)
    parser.add_argument("-N", "--negative_prompt", "--neg", dest='negative_prompts', nargs='+', action='append')
    parser.add_argument("-m", "--model", dest='models', nargs='+', action='append', required=True)
    parser.add_argument("-o", "--output", dest='output_dir', required=True, help="output directory")
    parser.add_argument("-s", "--sampler", dest='samplers', nargs='+', action='append')
    parser.add_argument("-n", "--num", dest='num_images', type=int, default=10, help="num images")
    parser.add_argument("--seed", dest='base_seed', type=int, default=0)
    parser.add_argument("--cfg", dest='cfgs', nargs='+', action='append', help="guidance scale")
    parser.add_argument("--batch", dest='batch_size', type=int, default=1, help="num images to generate in parallel")
    parser.add_argument("--width", dest="width", type=int, default=0)
    parser.add_argument("--height", dest="height", type=int, default=0)
    parser.add_argument("-f", dest="filename", help="read command line arguments from file") # dummy so the help shows this argument

    # support loading from a file with "-f filename". using parser.add_argument for
    # loading arguments is cumbersome and doesn't work great, so the above 
    # parser.add_argument("-f"...) is just to get the help text.
    config_args = []
    args = deque(sys.argv[1:])
    while len(args) > 0:
        arg = args.popleft()
        if arg == "-f":
            filename = args.popleft()
            with open(filename, "r") as file:
                config_args.extend(shlex.split(file.read(), comments=True))
            continue
        config_args.append(arg)

    config = parser.parse_args(config_args)
    if config.negative_prompts is None:
        config.negative_prompts = [[]]
    if config.cfgs is None:
        config.cfgs = [["7"]]
    if config.samplers is None:
        config.samplers = [["dpm++1:20"]]

    config.prompts = [inside for outside in config.prompts for inside in outside]
    config.negative_prompts = [inside for outside in config.negative_prompts for inside in outside]
    config.models = [inside for outside in config.models for inside in outside]
    config.samplers = [inside for outside in config.samplers for inside in outside]
    config.cfgs = [int(inside) for outside in config.cfgs for inside in outside]

    if len(config.negative_prompts) > 1 and len(config.negative_prompts) != len(config.prompts):
        raise Exception(f"got {len(config.prompts)} and {len(config.negative_prompts)}. negative must be 0, 1, or the same length as prompts")

    # save this for the logfile.
    config.config_args = config_args
    return config

if __name__ == "__main__":
    config = parse_args()
    image_gen = txt2img.ImageGenerator(config.batch_size)
    gen(image_gen, config)
