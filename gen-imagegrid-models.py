from collections import namedtuple
import os
import sys
import torch
from diffusers import StableDiffusionPipeline
from diffusers import EulerAncestralDiscreteScheduler, DDIMScheduler

MODELS_DIR = "/workspace/outputs"
IMAGES_DIR = "/workspace/output.images"

PROMPT = "a photo of alexhin person"
CFG = 4
NUM_IMAGES = 10
SEED = 0

schedulers = {
    'ddim': DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False),
    'euler_a': EulerAncestralDiscreteScheduler(),
}
SCHEDULER_NAME = 'euler_a'
#STEPS = 50
STEPS = 80
scheduler = schedulers[SCHEDULER_NAME]

Config = namedtuple("Config", "model_dir,model_name,steps,prompt")

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

            prompt = PROMPT
            if "w" in model_name:
                prompt = prompt.replace("alexhin person", "alexhin woman")
            config = Config(steps_dir, model_name, int(steps), prompt)
            res.append(config)

    return res

def gen_samples(config: Config):
    sample_dir = f"{IMAGES_DIR}/{config.model_name}_{int(config.steps):04}-{PROMPT}-{SCHEDULER_NAME}_{STEPS}"
    os.makedirs(sample_dir, exist_ok=True)
        
    pipe = StableDiffusionPipeline.from_pretrained(config.model_dir, revision="fp16", torch_dtype=torch.float16)
    pipe.to("cuda")

    for idx in range(NUM_IMAGES):
        filename = f"{sample_dir}/{idx:02}.{idx:02}.png"
        print(f"{idx + 1}/{NUM_IMAGES}: {filename}")
        if os.path.exists(filename):
            continue
        generator = torch.Generator("cuda").manual_seed(SEED + idx)
        images = pipe(config.prompt, guidance_scale=CFG, generator=generator, scheduler=scheduler, num_inference_steps=STEPS, safety_checker=None).images
        images[0].save(filename)
    print()

if __name__ == "__main__":
    configs = get_configs()
    for idx, config in enumerate(configs):
        print(f"\033[1;32m{idx + 1}/{len(configs)} gen_samples for {config.model_dir} ({config.model_name} : {config.steps})\033[0m")
        gen_samples(config)
        print()
    
    os.chdir(IMAGES_DIR)
    os.system(f"python /scripts/imagegrid.py {' '.join(sys.argv[1:])} > imagegrid.html")

