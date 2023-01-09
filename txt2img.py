from collections import namedtuple
from typing import Callable, List
import os
import sys
import torch
import PIL, PIL.Image, PIL.ImageDraw
import json
from PIL.PngImagePlugin import PngInfo
import importlib, importlib_metadata

from diffusers import DiffusionPipeline, StableDiffusionPipeline, StableDiffusionInpaintPipeline
from diffusers import DDIMScheduler, EulerDiscreteScheduler # works for SD2
from diffusers import EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler, KarrasVeScheduler, ScoreSdeVeScheduler # doesn't work for SD2

# from diffusers/examples/dreambooth/train-dreambooth.py
_xformers_available = importlib.util.find_spec("xformers") is not None
try:
    _xformers_version = importlib_metadata.version("xformers")
    _xformers_available = True
    print(f"Using xformers")
except importlib_metadata.PackageNotFoundError:
    _xformers_available = False
    print(f"xformers not available")

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
    negative_prompt: str
    num_images: int
    seed: int
    width: int
    height: int

    def __init__(self,
                 prompt: str, model_dir: str, 
                 negative_prompt: str = None,
                 model_str: str = "",
                 root_output_dir: str = ".", 
                 sampler_str: str = "dpm++1:20",
                 guidance_scale: float = 7, num_images: int = 1, seed: int = 0,
                 width: int = 0, height: int = 0):
        self.model_dir = model_dir

        if not model_str and model_dir:
            path_components = model_dir.split("/")
            last_component = path_components[-1].replace("checkpoint-", "").replace("save-", "").replace("epoch-", "")
            if len(path_components) >= 2 and all([c.isdigit() for c in last_component]):
                model_steps = last_component
                model_name = path_components[-2]
                model_str = f"{model_name}_{model_steps}"
            else:
                model_str = last_component

        self.model_str = model_str
        self.prompt = prompt
        self.negative_prompt = negative_prompt

        self.sampler_name, self.sampler_steps = sampler_str.split(":")
        self.sampler_steps = int(self.sampler_steps)
        self.guidance_scale = guidance_scale
        self.num_images = num_images
        self.seed = seed

        self.width = width
        self.height = height

        prompt_str = self.prompt
        if self.negative_prompt:
            prompt_str += f" || {negative_prompt}"

        self.output_dir = f"{root_output_dir}/{self.model_str}--{prompt_str}--{self.sampler_name}_{self.sampler_steps},c{self.guidance_scale:02}"
        if self.width:
            self.output_dir += f",width{self.width}"
        if self.height:
            self.output_dir += f",height{self.height}"

        if self.sampler_name not in SCHEDULERS:
            raise Exception(f"unknown scheduler '{self.sampler_name}'")

class ImageGenerator:
    pipeline = None
    scheduler = None

    last_sampler_name: str = ""
    last_model_dir: str = ""

    num_parallel: int = 0

    image_blank: PIL.Image = None
    image_mask: PIL.Image = None

    def __init__(self, num_parallel: int = 1):
        self.num_parallel = num_parallel

    def gen_images(self, image_set: ImageSet, 
                    save_image_fun: Callable[[ImageSet, int, str, PIL.Image.Image, PngInfo], None] = None) -> int:

        def _save_image(image_set: ImageSet, idx: int, filename: str, image: PIL.Image.Image, metadata: PngInfo):
            image.save(filename, pnginfo=metadata)
        
        inpainting = "inpainting" in image_set.model_str
        if inpainting and self.image_blank is None:
            self.image_blank = PIL.Image.new(mode="RGB", size=(512, 512))
            self.image_mask = PIL.Image.new(mode="RGB", size=(512, 512), color="white")

        if save_image_fun is None:
            save_image_fun = _save_image

        # figure out what output directories we need
        os.makedirs(image_set.output_dir, exist_ok=True)
        filenames = [f"{image_set.output_dir}/{idx + 1:02}.{image_set.seed + idx:010}.png"
                     for idx in range(image_set.num_images)]
        needed_filenames = [filename for filename in filenames if filename is None or not os.path.exists(filename)]
        print(f"\033[1;32m{image_set.output_dir}\033[0m: {len(needed_filenames)} to generate")
        if len(needed_filenames) == 0:
            return 0
        
        # re-create scheduler/pipeline only when the sampler or model changes.
        if image_set.model_dir != self.last_model_dir:
            if inpainting:
                self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(image_set.model_dir, revision="fp16", torch_dtype=torch.float16, safety_checker=None)
            else:
                self.pipeline = StableDiffusionPipeline.from_pretrained(image_set.model_dir, revision="fp16", torch_dtype=torch.float16, safety_checker=None)

            self.pipeline = self.pipeline.to("cuda")
            self.last_model_dir = image_set.model_dir

            if _xformers_available:
                self.pipeline.unet.enable_xformers_memory_efficient_attention()


        if image_set.sampler_name != self.last_sampler_name or self.pipeline.scheduler is None:
            scheduler_fun = SCHEDULERS[image_set.sampler_name]
            self.pipeline.scheduler = scheduler_fun(self.pipeline)
            self.last_sampler_name = image_set.sampler_name

        num_generated = len(needed_filenames)
        while len(needed_filenames) > 0:
            num_needed = len(needed_filenames)
            num_batch = min(num_needed, self.num_parallel)
            num_existing = image_set.num_images - num_needed
            print(f"{num_existing + 1}/{image_set.num_images}: {needed_filenames[0]}")

            kwargs = {}
            if image_set.width != 0:
                kwargs['width'] = image_set.width
            if image_set.height != 0:
                kwargs['height'] = image_set.height

            if inpainting:
                kwargs['image'] = self.image_blank
                kwargs['mask_image'] = self.image_mask

            seed = image_set.seed + num_existing
            generator = torch.Generator("cuda").manual_seed(seed)
            images: List[PIL.Image.Image] = \
                self.pipeline(image_set.prompt, 
                              negative_prompt=image_set.negative_prompt,
                              generator=generator,
                              guidance_scale=image_set.guidance_scale, 
                              num_inference_steps=image_set.sampler_steps,
                              num_images_per_prompt=num_batch,
                              **kwargs).images

            for idx in range(num_batch):
                info = {
                    'model_dir': image_set.model_dir,
                    'model_str': image_set.model_str,
                    'prompt': image_set.prompt,
                    'sampler_name': image_set.sampler_name,
                    'sampler_steps': str(image_set.sampler_steps),
                    'guidance_scale': str(image_set.guidance_scale),
                    'seed': str(image_set.seed + num_existing + idx)
                }

                # TODO: this is a bit of hack. these arguments are from gen-many, which
                # breaks the composition of 'gen-many uses txt2img' (but not vice-versa)
                cmdline = (f"-m {image_set.model_dir} "
                           f"--prompt '{image_set.prompt}' "
                           f"-s {image_set.sampler_name}:{image_set.sampler_steps} "
                           f"--cfg {image_set.guidance_scale} "
                           f"--seed {image_set.seed + num_existing + idx}")

                if image_set.negative_prompt:
                    info['negative_prompt'] = image_set.negative_prompt
                    cmdline += f" --negative_prompt '{image_set.negative_prompt}'"

                metadata = PngInfo()
                metadata.add_text('json', json.dumps(info))
                metadata.add_text('cmdline', cmdline)
                save_image_fun(image_set, idx, needed_filenames[idx], images[idx], metadata)
            
            needed_filenames = needed_filenames[num_batch:]

        print()
        return num_generated

if __name__ == "__main__":
    image_sets = []
    models = {
        'sd2': "stabilityai/stable-diffusion-2",
        'sd15': "runwayml/stable-diffusion-v1-5",
    }
    for model_name, dirname in models.items():
        for sampler_str in ['dpm++1:20']:
            for prompt in ['cute dog']:
                image_set = ImageSet(prompt, dirname, model_str=model_name,
                                     root_output_dir=dirname, 
                                     sampler_str=sampler_str,
                                     num_images=1)
                image_set.output_dir = "."
                image_sets.append(image_set)

    gen = ImageGenerator()
    for image_set in image_sets:
        gen.gen_images(image_set, save_image_fun=lambda iset, idx, filename, img, metadata: img.save(f"{idx}-{iset.model_str}.png", pnginfo=metadata))
