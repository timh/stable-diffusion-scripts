import os
import os.path
import re
import subprocess
import argparse
import shlex
import time
from collections import namedtuple
from typing import List

import txt2img
from txt2img import ImageSet

# ddim, k_dpm_2_a, k_dpm_2, k_euler_a, k_euler, k_heun, k_lms, plms

# generator for combinatorial set of images to generate
def gen_renders(config: argparse.Namespace):
    for model in config.models:
        for prompt in config.prompts:
            for sampler in config.samplers:
                for cfg in config.cfgs:
                    one = ImageSet(output_dir=config.output_dir, prompt=prompt, model_dir=model, sampler_str=sampler, guidance_scale=cfg,
                                    seed=config.base_seed, num_images=config.num_images)
                    yield one

def gen(image_gen: txt2img.ImageGenerator, config: argparse.Namespace):
    for one in gen_renders(config):
        time_start = time.perf_counter()
        image_gen.gen_images(one)
        time_end = time.perf_counter()

        time_filename = f"{one.output_dir}/timing.txt"
        with open(time_filename, "w") as file:
            file.write(f"{time_end - time_start}\n")

class LoadFromFile (argparse.Action):
    def __call__ (self, parser, config, values, option_string = None):
        with values as file:
            import copy

            old_actions = parser._actions
            old_required = {a.dest : a.required for a in old_actions}
            file_actions = copy.deepcopy(old_actions)

            # make none of the args required so we can get thru reading the file in
            # parser.parse_args, below.
            for act in file_actions:
                act.required = False

            parser._actions = file_actions
            parser.parse_args(shlex.split(file.read(), comments=True), config)

            # make any still-missing args required again, so we error out.
            for act in file_actions:
                if getattr(config, act.dest, None) is None:
                    act.required = old_required[act.dest]
            # parser._actions = old_actions

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="gen many sample images")
    parser.add_argument("-p", "--prompt", dest='prompts', nargs='+', action='append', required=True)
    parser.add_argument("-m", "--model", dest='models', nargs='+', action='append', required=True)
    parser.add_argument("-o", "--output", dest='output_dir', required=True, help="output directory")
    parser.add_argument("-s", "--sampler", dest='samplers', nargs='+', action='append')
    parser.add_argument("-n", "--num", dest='num_images', type=int, default=10, help="num images")
    parser.add_argument("--seed", dest='base_seed', type=int, default=0)
    parser.add_argument("--cfg", dest='cfgs', nargs='+', action='append', help="guidance scale")
    parser.add_argument("--batch", dest='batch_size', type=int, default=1, help="num images to generate in parallel")
    parser.add_argument("-f", "--filename", type=open, action=LoadFromFile)

    config = parser.parse_args()
    if config.cfgs is None:
        config.cfgs = [["7"]]
    if config.samplers is None:
        config.samplers = [["ddim:30"]]

    config.prompts = [inside for outside in config.prompts for inside in outside]
    config.models = [inside for outside in config.models for inside in outside]
    config.samplers = [inside for outside in config.samplers for inside in outside]
    config.cfgs = [int(inside) for outside in config.cfgs for inside in outside]

    return config

if __name__ == "__main__":
    config = parse_args()
    image_gen = txt2img.ImageGenerator(config.batch_size)
    gen(image_gen, config)
