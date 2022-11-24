from collections import namedtuple
from typing import Callable
import os
import sys
import torch
import PIL
from diffusers import StableDiffusionPipeline, VersatileDiffusionTextToImagePipeline
from diffusers import EulerAncestralDiscreteScheduler, DDIMScheduler, EulerDiscreteScheduler, DPMSolverMultistepScheduler, KarrasVeScheduler, ScoreSdeVeScheduler

SCHEDULERS = {
    'ddim': DDIMScheduler,                      # ddim:50 works for SD2
    'euler_a': EulerAncestralDiscreteScheduler, # euler_a:50 generates noise for SD2
    'euler': EulerDiscreteScheduler,            # euler:50 works
    'dpm': DPMSolverMultistepScheduler,         # dpm:50 generates noise
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
                 model_dir: str, model_str: str, prompt: str, 
                 output_dir: str = ".", sampler_name: str = "ddim", sampler_steps: int = 30,
                 guidance_scale: float = 7, num_images: int = 1, seed: int = -1):
        self.model_dir = model_dir
        self.model_str = model_str
        self.prompt = prompt

        self.output_dir = output_dir
        self.sampler_name = sampler_name
        self.sampler_steps = sampler_steps
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
            output_dir = f"{image_set.output_dir}/{image_set.model_str}-{image_set.prompt}-{image_set.sampler_name}_{image_set.sampler_steps}"
            os.makedirs(output_dir, exist_ok=True)
            filename = f"{output_dir}/{idx:10}.{idx + 1:02}.png"
            return filename

        def _save_image(image_set: ImageSet, idx: int, filename: str, image: PIL.Image.Image):
            image.save(filename)

        if filename_func is None:
            filename_func = _filename

        if save_image_fun is None:
            save_image_fun = _save_image

        # re-create scheduler/pipeline only when the sampler or model changes.
        if image_set.sampler_name != self.last_sampler_name:
            self.scheduler = image_set.scheduler_class.from_pretrained(image_set.model_dir, subfolder="scheduler")
            self.last_sampler_name = image_set.sampler_name
        if image_set.model_dir != self.last_model_dir:
            self.pipeline = StableDiffusionPipeline.from_pretrained(image_set.model_dir, revision="fp16", torch_dtype=torch.float16)
            self.pipeline = self.pipeline.to("cuda")
            self.last_model_dir = image_set.model_dir
        self.pipeline.scheduler = self.scheduler

        filenames = [filename_func(image_set, idx) for idx in range(image_set.num_images)]
        needed_filenames = [filename for filename in filenames if filename is None or not os.path.exists(filename)]

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
                                    num_images_per_prompt=num_batch,
                                    safety_checker=None).images

            for idx in range(num_batch):
                save_image_fun(image_set, idx, needed_filenames[idx], images[idx])
            
            needed_filenames = needed_filenames[num_batch:]

        print()

if __name__ == "__main__":
    image_sets = []
    for model_name in ['stable-diffusion-2', 'stable-diffusion-v1-5']:
        dirname = f"/home/tim/devel/{model_name}"

        for sampler in ['euler', 'ddim']:
            image_set = ImageSet(dirname, model_name, "photo of a dog sitting on a table", 
                                sampler_name=sampler, sampler_steps=30,
                                num_images=7)
            image_sets.append(image_set)


    gen = ImageGenerator()
    for image_set in image_sets:
        gen.gen_images(image_set)
