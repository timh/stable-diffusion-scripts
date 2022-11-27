from collections import namedtuple
from typing import Callable
import os
import sys
import torch
import PIL
from diffusers import StableDiffusionPipeline
from diffusers import DDIMScheduler, EulerDiscreteScheduler # works for SD2
from diffusers import EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler, KarrasVeScheduler, ScoreSdeVeScheduler # doesn't work for SD2

SCHEDULERS = {
    'ddim': DDIMScheduler,                      # ddim:50 works for SD2
    'euler': EulerDiscreteScheduler,            # euler:50 works for SD2
    'euler_a': EulerAncestralDiscreteScheduler, # BUG euler_a:50 generates noise for SD2
    'dpm': DPMSolverMultistepScheduler,         # BUG dpm:50 generates noise for SD2
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
                 output_dir: str = ".", 
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

        self.output_dir = output_dir
        self.sampler_name, self.sampler_steps = sampler_str.split(":")
        self.sampler_steps = int(self.sampler_steps)
        self.guidance_scale = guidance_scale
        self.num_images = num_images
        self.seed = seed

        if self.sampler_name not in SCHEDULERS:
            raise Exception(f"unknown scheduler '{self.scheduler_name}'")
        self.scheduler_class = SCHEDULERS[self.sampler_name]()

class ImageGenerator:
    pipeline = None
    scheduler = None

    last_sampler_name: str = ""
    last_model_dir: str = ""

    num_parallel: int = 0

    def __init__(self, num_parallel: int = 1):
        self.num_parallel = num_parallel

    def gen_images(self, image_set: ImageSet, 
                    filename_func: Callable[[ImageSet, int], str] = None,
                    save_image_fun: Callable[[ImageSet, int, str, PIL.Image.Image], None] = None):

        def _filename(image_set: ImageSet, idx: int) -> str:
            output_dir = f"{image_set.output_dir}/{image_set.model_str}--{image_set.prompt}--{image_set.sampler_name}_{image_set.sampler_steps},c{image_set.guidance_scale:02}"
            os.makedirs(output_dir, exist_ok=True)
            filename = f"{output_dir}/{idx:010}.{idx + 1:02}.png"
            return filename

        def _save_image(image_set: ImageSet, idx: int, filename: str, image: PIL.Image.Image):
            image.save(filename)

        if filename_func is None:
            filename_func = _filename

        if save_image_fun is None:
            save_image_fun = _save_image

        filenames = [filename_func(image_set, idx) for idx in range(image_set.num_images)]
        needed_filenames = [filename for filename in filenames if filename is None or not os.path.exists(filename)]
        if len(needed_filenames) == 0:
            return
        
        if needed_filenames[0] is not None:
            output_dir = os.path.dirname(needed_filenames[0])
            output_dir = output_dir.split('/')[-1]
        else:
            output_dir = ""
        print(f"\033[1;32m{output_dir}\033[0m: {len(needed_filenames)} to generate")

        # re-create scheduler/pipeline only when the sampler or model changes.
        if image_set.sampler_name != self.last_sampler_name or image_set.model_dir != self.last_model_dir:
            self.scheduler = image_set.scheduler_class.from_pretrained(image_set.model_dir, subfolder="scheduler")
            self.last_sampler_name = image_set.sampler_name
        if image_set.model_dir != self.last_model_dir:
            self.pipeline = StableDiffusionPipeline.from_pretrained(image_set.model_dir, revision="fp16", torch_dtype=torch.float16, safety_checker=None)
            self.pipeline = self.pipeline.to("cuda")
            self.last_model_dir = image_set.model_dir
        self.pipeline.scheduler = self.scheduler

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
                image_set = ImageSet(prompt, dirname, model_str=model_name,
                                     sampler_str=sampler_str,
                                     num_images=1)
                image_sets.append(image_set)


    gen = ImageGenerator()
    for image_set in image_sets:
        gen.gen_images(image_set)
