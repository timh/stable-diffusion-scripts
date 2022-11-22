from collections import namedtuple
import os
import torch
from diffusers import StableDiffusionPipeline
from diffusers import EulerAncestralDiscreteScheduler, DDIMScheduler

MODELS_DIR = "/workspace/outputs"
IMAGES_DIR = "/workspace/output.images"

PROMPT = "a photo of alexhin"
CFG = 4
NUM_IMAGES = 10
SEED = 0
STEPS = 50

Config = namedtuple("Config", "model_dir,model_name,steps")

def get_configs():
    res = []

    # generate directories & filenames of the form that imagegrid looks for..
    for model_name in os.listdir(MODELS_DIR):
        model_dir = f"{MODELS_DIR}/{model_name}"
        dir0 = f"{model_dir}/0"
        if not os.path.isdir(dir0):
            continue

        for steps in os.listdir(model_dir):
            steps_dir = f"{model_dir}/{steps}"

            if not os.path.isdir(steps_dir) or steps == "0":
                continue
            if not all(lambda c: c.isdigit() for c in steps):
                continue

            config = Config(steps_dir, model_name, int(steps))
            res.append(config)

    return res

def gen_samples(config: Config):
    sample_dir = f"{IMAGES_DIR}/{config.model_name}_{int(config.steps):04}-sample-ddim_50"
    os.makedirs(sample_dir, exist_ok=True)
        
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)

    pipe = StableDiffusionPipeline.from_pretrained(config.model_dir, revision="fp16", torch_dtype=torch.float16)
    pipe.to("cuda")

    for idx in range(NUM_IMAGES):
        filename = f"{sample_dir}/{idx:02}.{idx:02}.png"
        print(f"{idx + 1}/{NUM_IMAGES}: {filename}")
        if os.path.exists(filename):
            continue
        generator = torch.Generator("cuda").manual_seed(SEED + idx)
        images = pipe(PROMPT, guidance_scale=CFG, generator=generator, num_inference_steps=STEPS, scheduler=scheduler, safety_checker=None).images
        images[0].save(filename)
    print()

if __name__ == "__main__":
    configs = get_configs()
    for config in configs:
        print(f"\033[1;32mgen_samples for {config.model_dir} ({config.model_name} : {config.steps})\033[0m")
        gen_samples(config)
        print()

