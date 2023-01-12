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
PROMPT = "portrait of alexhin, dramatic lighting, tone mapped, elegant, digital painting, artstation, smooth, sharp focus"
BASE_SEED = 0

@torch.no_grad()
def gen(
    pipe: StableDiffusionPipeline,
    prompt: Union[str, List[str]],
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    callback_steps: Optional[int] = 1,
):
        # 0. Default height and width to unet
        height = height or pipe.unet.config.sample_size * pipe.vae_scale_factor
        width = width or pipe.unet.config.sample_size * pipe.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        pipe.check_inputs(prompt, height, width, callback_steps)

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = pipe._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # START TIM
        text_inputs = pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = pipe.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if hasattr(pipe.text_encoder.config, "use_attention_mask") and pipe.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None
        
        text_embeddings = pipe.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        manual_embeddings = torch.zeros(text_embeddings.size(), dtype=text_embeddings.dtype).to(device)
        for idx, input_id in enumerate(text_input_ids):
            idx_input_ids = torch.zeros(text_input_ids.size(), dtype=torch.int32).to(device)
            idx_input_ids[idx] = input_id
            input_embeddings = pipe.text_encoder(idx_input_ids)[0]
            # if idx > 10:
            #     input_embeddings = input_embeddings.mul(1.2)
            manual_embeddings[idx] = input_embeddings[0] # * 1.2

        builtin_embeddings = pipe._encode_prompt(prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt)
        uncond_builtin = builtin_embeddings[0:1]
        text_embeddings = torch.cat([uncond_builtin, manual_embeddings])

        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # END TIM
        
        # 4. Prepare timesteps
        pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = pipe.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = pipe.unet.in_channels
        latents = pipe.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * pipe.scheduler.order
        with pipe.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # 8. Post-processing
        image = pipe.decode_latents(latents)

        # 9. Run safety checker
        image, has_nsfw_concept = pipe.run_safety_checker(image, device, text_embeddings.dtype)

        # 10. Convert to PIL
        image = pipe.numpy_to_pil(image)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)


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
    my_image = gen(pipe, PROMPT, generator=generator, num_inference_steps=30, guidance_scale=7).images[0]

    generator = torch.Generator("cuda").manual_seed(BASE_SEED)
    standard_image = pipe(PROMPT, generator=generator, num_inference_steps=30, guidance_scale=7).images[0]

    overall_image.paste(standard_image, (0, 0))
    overall_image.paste(my_image, (SIZE, 0))

    overall_image.save("temp.png")
    os.system(f"mv temp.png output.png")
