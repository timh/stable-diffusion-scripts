import os
import os.path
import re
import subprocess
import argparse
from collections import namedtuple
from typing import List

# ddim, k_dpm_2_a, k_dpm_2, k_euler_a, k_euler, k_heun, k_lms, plms

OneRender = namedtuple('OneRender', 'model,prompt,sampler_type,sampler_steps,cfg,seed')

# generator for combinatorial set of images to generate
def gen_renders(config: argparse.Namespace):
    for model in config.models:
        for prompt in config.prompts:
            for sampler in config.samplers:
                sampler_type, sampler_steps = sampler.split(":")
                for cfg in config.cfgs:
                    for idx in range(config.num_images):
                        seed = config.base_seed + idx
                        one = OneRender(model=model, prompt=prompt, sampler_type=sampler_type, sampler_steps=sampler_steps, cfg=cfg, seed=seed)
                        yield one

def gen(config: argparse.Namespace):
    last_model: str = None
    last_cmd: List[str] = None
    last_outdir: str = None
    proc: subprocess.Popen = None

    def close_proc(last_cmd: List[str], proc: subprocess.Popen):
        if proc is not None:
            proc.stdin.close()
            ret = proc.wait()
            if ret != 0:
                raise Exception(f"{' '.join(last_cmd)} returned {ret}")

    for one in gen_renders(config):
        sampler_tag = f"{one.sampler_type}_{one.sampler_steps}"
        outdir = f"outputs/{one.model}-{one.prompt}-{sampler_tag} c{one.cfg}"
        if os.path.isdir(outdir):
            if last_outdir != outdir:
                print(f"\"{outdir}\" already exists, skipping.")
            continue
        last_outdir = outdir

        if last_model is None or last_model != one.model:
            close_proc(last_cmd, proc)
            cmd = [
                "accelerate", "launch",
                "scripts/invoke.py",
                "--model", one.model,
                "--from_file", "-"
            ]
            print(f"RUN: {' '.join(cmd)}")
            if not config.dry_run:
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            last_model = one.model
            last_cmd = cmd

        prompt = one.prompt
        if one.model in ["alexhin20_1e6_3500", "alex20_03500"]:
            prompt = prompt.replace("alexhin", "alexhin person")
        img_cmd = (f"{prompt} "
                   f"--sampler {one.sampler_type} --steps {one.sampler_steps} "
                   f"--cfg_scale {one.cfg} "
                   f"--seed {one.seed} "
                   f"--outdir \"{outdir}\"\n")
        print(f"img_cmd: {img_cmd}", end="")
        if not config.dry_run:
            proc.stdin.write(bytes(img_cmd, "utf-8"))
    
    close_proc(last_cmd, proc)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="gen many sample images", fromfile_prefix_chars="@")
    parser.add_argument("-p", "--prompt", dest='prompts', nargs='+', action='append', required=True)
    parser.add_argument("-m", "--model", dest='models', nargs='+', action='append', required=True)
    parser.add_argument("-s", "--sampler", dest='samplers', nargs='+', action='append')
    parser.add_argument("-n", "--num", dest='num_images', type=int, default=10, help="num images")
    parser.add_argument("--seed", dest='base_seed', type=int, default=0)
    parser.add_argument("--cfg", dest='cfgs', nargs='+', action='append', help="CFG")
    parser.add_argument("--dry-run", action='store_true')

    config = parser.parse_args()
    if config.cfgs is None:
        config.cfgs = [["10"]]
    if config.samplers is None:
        config.samplers = [["ddim:30"]]

    config.prompts = [inside for outside in config.prompts for inside in outside]
    config.models = [inside for outside in config.models for inside in outside]
    config.samplers = [inside for outside in config.samplers for inside in outside]
    config.cfgs = [inside for outside in config.cfgs for inside in outside]

    print(f"config = {config}")
    return config


if __name__ == "__main__":
    config = parse_args()
    gen(config)
