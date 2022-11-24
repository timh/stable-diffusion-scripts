from collections import namedtuple
from typing import Callable
import os
import sys
import torch
import PIL
# from diffusers.pipelines import StableDiffusionPipeline
# from diffusers.schedulers import EulerAncestralDiscreteScheduler, DDIMScheduler, EulerDiscreteScheduler, DPMSolverMultistepScheduler, KarrasVeScheduler, ScoreSdeVeScheduler
from diffusers import StableDiffusionPipeline
from diffusers import EulerAncestralDiscreteScheduler, DDIMScheduler, EulerDiscreteScheduler, DPMSolverMultistepScheduler, KarrasVeScheduler, ScoreSdeVeScheduler

SCHEDULERS = {
    # 'ddim': DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False),
    # 'euler_a': EulerAncestralDiscreteScheduler(),
    # 'euler': EulerDiscreteScheduler(),
    # 'lms': LMSDiscreteScheduler(),
    # 'dpm': DPMSolverMultistepScheduler(),
    # 'karras_ve': KarrasVeScheduler(),
    # 'sde_ve': ScoreSdeVeScheduler()
    'ddim': DDIMScheduler,
    'euler_a': EulerAncestralDiscreteScheduler,
    'euler': EulerDiscreteScheduler,
    'dpm': DPMSolverMultistepScheduler,
    'sde_ve': ScoreSdeVeScheduler,
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
        # self.scheduler = SCHEDULERS[self.sampler_name](self.sampler_steps)
        self.scheduler = SCHEDULERS[self.sampler_name](self.sampler_steps)

class ImageGenerator:
    pipeline = None
    pipeline_model_str: str = ""

    def gen_images(self, image_set: ImageSet, 
                    filename_func: Callable[[ImageSet, int], str] = None,
                    save_image_fun: Callable[[ImageSet, int, str, PIL.Image.Image], None] = None):

        def _filename(image_set: ImageSet, idx: int) -> str:
            output_dir = f"{image_set.output_dir}/{image_set.model_str}-{image_set.prompt}-{image_set.sampler_name}_{image_set.sampler_steps}"
            os.makedirs(output_dir, exist_ok=True)
            filename = f"{output_dir}/{idx:02}.{idx:02}.png"
            return filename

        def _save_image(image_set: ImageSet, idx: int, filename: str, image: PIL.Image.Image):
            image.save(filename)

        if filename_func is None:
            filename_func = _filename

        if save_image_fun is None:
            save_image_fun = _save_image


        pipe = None        
        for idx in range(image_set.num_images):
            filename = filename_func(image_set, idx)
            print(f"{idx + 1}/{image_set.num_images}: {filename}")
            if filename is not None and os.path.exists(filename):
                continue

            # if self.pipeline is None or self.pipeline_model_str != image_set.model_str:
            #     self.pipeline = StableDiffusionPipeline.from_pretrained(image_set.model_dir, revision="fp16", torch_dtype=torch.float16, safety_checker=None)
            #     self.pipeline.to("cuda")
            #     self.pipeline_model_str = image_set.model_str
            
            # pipeline = StableDiffusionPipeline.from_pretrained(image_set.model_dir, scheduler=image_set.scheduler, revision="fp16", torch_dtype=torch.float16, safety_checker=None)
            self.pipeline = StableDiffusionPipeline.from_pretrained(image_set.model_dir, scheduler=image_set.scheduler, safety_checker=None)
            self.pipeline.to("cuda")

            generator = torch.Generator("cuda").manual_seed(image_set.seed + idx)
            images = self.pipeline(image_set.prompt, 
                                   guidance_scale=image_set.guidance_scale, generator=generator, 
                                   scheduler=image_set.scheduler, num_inference_steps=image_set.sampler_steps,
                                   safety_checker=None).images

            save_image_fun(image_set, idx, filename, images[0])

        print()

if __name__ == "__main__":
    # im15 = ImageSet("/home/tim/devel/stable-diffusion-v1-5", "stable-diffusion-1.5", "a cat sitting on a table", sampler_steps=50)
    # im15inp = ImageSet("/home/tim/devel/stable-diffusion-inpainting", "stable-diffusion-1.5", "a cat sitting on a table", sampler_steps=50)

    image_sets = []
    # for steps in [2500, 3000, 3500]:
    #     model_name = "alexhin20v2pf_cos0.8e6_r12"
    #     dirname = f"/home/tim/Downloads/{model_name}/{steps}"
    #     model_str = f"{model_name}_{steps}"

    #     image_set = ImageSet(dirname, model_str, "portrait of alexhin person, pencil sketch", 
    #                          sampler_name='dpm', sampler_steps=50,
    #                          num_images=10)
    #     image_sets.append(image_set)
    for model_name in ['stable-diffusion-2', 'stable-diffusion-v1-5']:
        dirname = f"/home/tim/devel/{model_name}"

        for sampler in ['euler']:
            image_set = ImageSet(dirname, model_name, "photo of a dog sitting on a table", 
                                sampler_name=sampler, sampler_steps=50,
                                num_images=5)
            image_sets.append(image_set)


    gen = ImageGenerator()
    for image_set in image_sets:
        gen.gen_images(image_set)
