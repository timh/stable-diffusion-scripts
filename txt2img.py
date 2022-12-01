from collections import namedtuple
from typing import Callable
import os
import sys
import torch
import PIL
from diffusers import DiffusionPipeline, StableDiffusionPipeline
from diffusers import DDIMScheduler, EulerDiscreteScheduler # works for SD2
from diffusers import EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler, KarrasVeScheduler, ScoreSdeVeScheduler # doesn't work for SD2

# -- from diffusers docs:
# from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
# import torch
#
# repo_id = "stabilityai/stable-diffusion-2-base"
# pipe = DiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch.float16, revision="fp16")
#
# pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
# pipe = pipe.to("cuda")
#
# prompt = "High quality photo of an astronaut riding a horse in space"
# image = pipe(prompt, num_inference_steps=25).images[0]
# image.save("astronaut.png")
SCHEDULERS = {
       'ddim': lambda pipe: DDIMScheduler.from_config(pipe.scheduler.config),
      'euler': lambda pipe: EulerDiscreteScheduler.from_config(pipe.scheduler.config),

    'euler_a': lambda pipe: EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config),
        'dpm': lambda pipe: DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type='dpmsolver'),
       'dpm1': lambda pipe: DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type='dpmsolver', solver_order=1),
       'dpm2': lambda pipe: DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type='dpmsolver', solver_order=2),
     'dpm++1': lambda pipe: DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type='dpmsolver++', solver_order=1),
     'dpm++2': lambda pipe: DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type='dpmsolver++', solver_order=2),
}



class ImageSet:
    model_dir: str
    model_str: str
    # model_name: str
    # model_steps: int
    # model_seed: int
    output_dir: str
    sampler_name: str
    sampler_steps: int
    guidance_scale: float
    prompt: str
    num_images: int
    seed: int

    def __init__(self,
                 prompt: str, model_dir: str, model_str: str = "",
                 root_output_dir: str = ".", 
                 sampler_str: str = "ddim:30",
                 guidance_scale: float = 7, num_images: int = 1, seed: int = -1):
        self.model_dir = model_dir

        if not model_str and model_dir:
            path_components = model_dir.split("/")
            last_component = path_components[-1]
            if len(path_components) >= 2 and all([c.isdigit() for c in last_component]):
                model_steps = last_component
                model_name = path_components[-2]
                model_str = f"{model_name}_{model_steps}"
            else:
                model_str = last_component

        self.model_str = model_str
        self.prompt = prompt

        self.sampler_name, self.sampler_steps = sampler_str.split(":")
        self.sampler_steps = int(self.sampler_steps)
        self.guidance_scale = guidance_scale
        self.num_images = num_images
        self.seed = seed

        self.output_dir = f"{root_output_dir}/{self.model_str}--{self.prompt}--{self.sampler_name}_{self.sampler_steps},c{self.guidance_scale:02}"

        if self.sampler_name not in SCHEDULERS:
            raise Exception(f"unknown scheduler '{self.sampler_name}'")

class ImageGenerator:
    pipeline = None
    scheduler = None

    last_sampler_name: str = ""
    last_model_dir: str = ""

    num_parallel: int = 0

    def __init__(self, num_parallel: int = 1):
        self.num_parallel = num_parallel

    def gen_images(self, image_set: ImageSet, 
                    save_image_fun: Callable[[ImageSet, int, str, PIL.Image.Image], None] = None):

        def _save_image(image_set: ImageSet, idx: int, filename: str, image: PIL.Image.Image):
            image.save(filename)

        if save_image_fun is None:
            save_image_fun = _save_image

        # figure out what output directories we need
        os.makedirs(image_set.output_dir, exist_ok=True)
        filenames = [f"{image_set.output_dir}/{idx + 1:02}.{image_set.seed + idx:010}.png"
                     for idx in range(image_set.num_images)]
        needed_filenames = [filename for filename in filenames if filename is None or not os.path.exists(filename)]
        print(f"\033[1;32m{image_set.output_dir}\033[0m: {len(needed_filenames)} to generate")
        if len(needed_filenames) == 0:
            return
        
        # re-create scheduler/pipeline only when the sampler or model changes.
        if image_set.model_dir != self.last_model_dir:
            self.pipeline = StableDiffusionPipeline.from_pretrained(image_set.model_dir, revision="fp16", torch_dtype=torch.float16, safety_checker=None)
            self.pipeline = self.pipeline.to("cuda")

        if image_set.sampler_name != self.last_sampler_name or self.pipeline.scheduler is None:
            scheduler_fun = SCHEDULERS[image_set.sampler_name]
            self.pipeline.scheduler = scheduler_fun(self.pipeline)
            self.last_sampler_name = image_set.sampler_name

        while len(needed_filenames) > 0:
            num_needed = len(needed_filenames)
            num_batch = min(num_needed, self.num_parallel)
            num_existing = image_set.num_images - num_needed
            print(f"{num_existing + 1}/{image_set.num_images}: {needed_filenames[0]}")

            seed = image_set.seed + num_existing
            generator = torch.Generator("cuda").manual_seed(seed)
            images = self.pipeline(image_set.prompt, 
                                   generator=generator,
                                   guidance_scale=image_set.guidance_scale, 
                                   num_inference_steps=image_set.sampler_steps,
                                   num_images_per_prompt=num_batch).images

            for idx in range(num_batch):
                save_image_fun(image_set, idx, needed_filenames[idx], images[idx])
            
            needed_filenames = needed_filenames[num_batch:]

        print()

if __name__ == "__main__":
    image_sets = []
    models = {
        'sd2': "/home/tim/devel/stable-diffusion-2",
        'sd15': "/home/tim/devel/stable-diffusion-v1-5",
    }
    for model_name, dirname in models.items():
        for sampler_str in ['euler:50']:
            for prompt in ["photo of a cute dog", "color pencil sketch of a cute dog"]:
                image_set = ImageSet(prompt, root_output_dir=dirname, model_str=model_name,
                                     sampler_str=sampler_str,
                                     num_images=1)
                image_sets.append(image_set)


    gen = ImageGenerator()
    for image_set in image_sets:
        gen.gen_images(image_set)
