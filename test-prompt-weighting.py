from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from PIL import Image, ImageFont, ImageDraw
from fonts.ttf import Roboto

from collections import namedtuple, deque
from typing import List, Deque, Dict, Optional, Union, Callable
import torch, torch.nn.functional
import textwrap
import os, sys, re
import math
from pathlib import Path
import datetime

# MODEL = "/home/tim/models/stable-diffusion-v1-5+vae"
# MODEL = "/home/tim/models/ppp"
MODEL = "/home/tim/models/alex66-kohyanative-modelshoot-batch11@0.5_r2/epoch-000700"

SIZE = 512
# PROMPT = "portrait of alexhin, pencil sketch"
PROMPT = "portrait photo of alexhin wearing a blue hat"
BASE_SEED = 1

def encode_part(self: StableDiffusionPipeline, prompt: str, device: torch.device):
    text_inputs = self.tokenizer(
        prompt,
        #padding="max_length",
        #max_length=self.tokenizer.model_max_length,
        #truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
        removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
        print(
            "The following part of your input was truncated because CLIP can only handle sequences up to"
            f" {self.tokenizer.model_max_length} tokens: {removed_text}"
        )

    if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
        attention_mask = text_inputs.attention_mask.to(device)
    else:
        attention_mask = None

    text_embeddings = self.text_encoder(
        text_input_ids.to(device),
        attention_mask=attention_mask,
    )
    text_embeddings = text_embeddings[0]

    print(f"prompt '{prompt}'")
    print(f"  text_input_ids size {text_input_ids.size()}")
    print(f"  text_input_ids {text_input_ids}")
    print(f"  part_embeddings size {text_embeddings.size()}")

    return text_embeddings

def _encode_prompt_fun(self: StableDiffusionPipeline):
    def fun(*args, **kwargs):
        return _encode_prompt(self, *args, **kwargs)
    return fun

def _encode_prompt(self: StableDiffusionPipeline, prompt: str, device: torch.device, num_images_per_prompt: int, do_classifier_free_guidance: bool, negative_prompt: str):
    r"""
    Encodes the prompt into text encoder hidden states.

    Args:
        prompt (`str` or `list(int)`):
            prompt to be encoded
        device: (`torch.device`):
            torch device
        num_images_per_prompt (`int`):
            number of images that should be generated per prompt
        do_classifier_free_guidance (`bool`):
            whether to use classifier free guidance or not
        negative_prompt (`str` or `List[str]`):
            The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
            if `guidance_scale` is less than `1`).
    """
    batch_size = len(prompt) if isinstance(prompt, list) else 1

    ##
    ## TIM HACKS
    ##
    text_embeddings: torch.Tensor = None
    out_idx = 1
    for part_idx, part in enumerate(prompt.split(",")):
        part = part.strip()
        if part_idx > 0:
            part = ", " + part
        part_embeddings = encode_part(self, part, device)
        if text_embeddings is None:
            text_embeddings = torch.zeros(torch.Size([1, self.tokenizer.model_max_length, part_embeddings.shape[-1]]), dtype=part_embeddings.dtype).to(device)
        num_embeddings = part_embeddings.shape[1]

        # copy the embeddings, excluding start and end.
        for idx in range(num_embeddings - 2):
            text_embeddings[0][out_idx + idx] = part_embeddings[0][idx + 1]
        out_idx += num_embeddings

    # now add start & end embeddings.    
    special_input_ids = self.tokenizer("", return_tensors="pt").input_ids
    print(f"special_input_ids = {special_input_ids}")
    special_embedding = self.text_encoder(special_input_ids.to(device))[0]
    text_embeddings[0][0] = special_embedding[0][0]
    # for idx in range(out_idx, self.tokenizer.model_max_length):
    #     text_embeddings[0][idx] = special_embedding[0][1]
    text_embeddings[0][out_idx] = special_embedding[0][1]
    ##
    ## END TIM HACKS
    ##

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    bs_embed, seq_len, _ = text_embeddings.shape
    text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
    text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

    # get unconditional embeddings for classifier free guidance
    if do_classifier_free_guidance:
        uncond_tokens: List[str]
        if negative_prompt is None:
            uncond_tokens = [""] * batch_size
        elif type(prompt) is not type(negative_prompt):
            raise TypeError(
                f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                f" {type(prompt)}."
            )
        elif isinstance(negative_prompt, str):
            uncond_tokens = [negative_prompt]
        elif batch_size != len(negative_prompt):
            raise ValueError(
                f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                " the batch size of `prompt`."
            )
        else:
            uncond_tokens = negative_prompt

        uncond_input = self.tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=text_embeddings.shape[-2],
            truncation=True,
            return_tensors="pt",
        )
        print(f"uncond_input.size() {uncond_input.input_ids.size()}")

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = uncond_input.attention_mask.to(device)
        else:
            attention_mask = None

        uncond_embeddings = self.text_encoder(
            uncond_input.input_ids.to(device),
            attention_mask=attention_mask,
        )
        uncond_embeddings = uncond_embeddings[0]
        print(f"uncond_embeddings.size {uncond_embeddings.size()}")

        # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
        seq_len = uncond_embeddings.shape[1]
        uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_prompt, 1)
        uncond_embeddings = uncond_embeddings.view(batch_size * num_images_per_prompt, seq_len, -1)

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    print(f"text_embeddings.size {text_embeddings.size()}")
    return text_embeddings

def mem(s: str):
    free, used = torch.cuda.mem_get_info()
    print(f"{s}: free {free/1024/1024}")


if __name__ == "__main__":
    pipe = StableDiffusionPipeline.from_pretrained(MODEL, torch_dtype=torch.float16, safety_checker=None).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type='dpmsolver++', solver_order=1)

    width = SIZE * 2
    height = SIZE
    overall_image = Image.new(mode="RGB", size=[width, height])
    draw = ImageDraw.Draw(overall_image)
    font = ImageFont.truetype(Roboto, 30)

    generator = torch.Generator("cuda").manual_seed(BASE_SEED)
    standard_image = pipe(PROMPT, generator=generator, num_inference_steps=30, guidance_scale=7).images[0]

    generator = torch.Generator("cuda").manual_seed(BASE_SEED)
    pipe._encode_prompt = _encode_prompt_fun(pipe)
    my_image = pipe(PROMPT, generator=generator, num_inference_steps=30, guidance_scale=7).images[0]

    overall_image.paste(standard_image, (0, 0))
    overall_image.paste(my_image, (SIZE, 0))

    overall_image.save("temp.png")
    os.system(f"mv temp.png output.png")
