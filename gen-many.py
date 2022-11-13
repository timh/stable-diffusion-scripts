import os
import os.path
import re
import subprocess

# General rendering:
#   --prompt PROMPT       prompt string
#   -s STEPS, --steps STEPS
#                         Number of steps
#   -S SEED, --seed SEED  Image seed; a +ve integer, or use -1 for the previous seed, -2 for the one before that, etc
#   -n ITERATIONS, --iterations ITERATIONS
#                         Number of samplings to perform (slower, but will provide seeds for individual images)
#   -W WIDTH, --width WIDTH
#                         Image width, multiple of 64
#   -H HEIGHT, --height HEIGHT
#                         Image height, multiple of 64
#   -C CFG_SCALE, --cfg_scale CFG_SCALE
#                         Classifier free guidance (CFG) scale - higher numbers cause generator to "try" harder.

def gen_one_set(model_name:str, sampler_str:str, prompt:str, base_seed:int, num_imgs:int):
    sampler, sampler_steps = sampler_str.split(":")
    sampler_steps = int(sampler_steps)
    print(f"gen_one: model {model_name}, sampler {sampler}, steps {sampler_steps}, prompt \"{prompt}\", base_seed {base_seed}, num_imgs {num_imgs}")

    sampler_tag = f"{sampler}_{sampler_steps}"
    # prompt_str = prompt.replace(" ", "_")
    prompt_str = prompt
    outdir = f"outputs/{model_name}-{prompt_str}-{sampler_tag}"
    if os.path.isdir(outdir):
        print(f"skipping {outdir}")
        return
    
    cmd = [
        "python", "scripts/invoke.py",
        "--outdir", outdir,
        "--sampler", sampler,
        "--model", model_name,
        "--from_file", "-"
    ]

    print(f"RUN: {cmd}")
    proc = subprocess.Popen(cmd, close_fds=True, stdin=subprocess.PIPE)
    if True:
        print("subproc")
        seed = base_seed
        for i in range(num_imgs):
            img_cmd = f"{prompt} -s {sampler_steps} -C 12 -S {seed}\n"
            print(f"  write: {img_cmd}")
            proc.stdin.write(bytes(img_cmd, "utf-8"))
            seed = seed + 1
        proc.stdin.close()
        ret = proc.wait()
        if ret != 0:
            raise Exception(f"{cmd} returned {ret}")

def gen(model_names, sampler_strings, prompts, base_seed, num_imgs):
    if isinstance(model_names, list):
        for model_name in model_names:
            gen(model_name, sampler_strings, prompts, base_seed, num_imgs)
        return

    if isinstance(prompts, list):
        for prompt in prompts:
            gen(model_names, sampler_strings, prompt, base_seed, num_imgs)
        return

    if isinstance(sampler_strings, list):
        for sampler_str in sampler_strings:
            gen(model_names, sampler_str, prompts, base_seed, num_imgs)
        return
    
    gen_one_set(model_names, sampler_strings, prompts, base_seed, num_imgs)
    

#ddim, k_dpm_2_a, k_dpm_2, k_euler_a, k_euler, k_heun, k_lms, plms
#samplers = ["ddim:50", "k_dpm_2:50"]
#samplers = ["k_dpm_2:20", "k_dpm_2:50", "k_euler:20", "k_euler:50", "k_euler_a:20", "k_euler_a:50"]
samplers = ["k_dpm_2:20"]

#models = "alex20_03500 alex20_06000 alex34_03500 alex34_04000 alex34_04500 alex34_05000 alex34_05500 alex34_06000 alex34_06500 alex34_07000 alex34_07500 alex34_08000".split(" ")
#models = "alex34_04000 alex34_04500 alex34_05000 alex34_05500 alex34_06500 alex34_07500".split(" ")
#models = "alex20_03500 alex34_03500 alex34_04000 alex34_05000 alex34_06000 alex34_06500".split(" ")
#models = "alex20_03500"
# models = (#[f"alexhin20-inpainting2_{num:05}" for num in range(3000, 19000+1, 4000)] +
#           [f"alexhin20-inpainting3_{num:05}" for num in range(1000, 16000+1, 1000)])
models = ("alexhin24-inpainting_05_1000 alexhin24-inpainting_05_1500 alexhin24-inpainting_05_2000 " +
          "alexhin24-inpainting_10_1000 alexhin24-inpainting_10_1500 alexhin24-inpainting_10_2000").split(" ")

base_seed = 0
num_imgs = 20
# for alexhin20-inpainting2
# prompts = [ "portrait of alexhin woman, dramatic lighting, tone mapped, elegant, digital painting, artstation, smooth, sharp focus",
#             "portrait of alexhin woman, pencil sketch" ]
# for alexhin20-inpainting3
prompts = [ "portrait of alexhin person, dramatic lighting, tone mapped, elegant, digital painting, artstation, smooth, sharp focus",
            "portrait of alexhin person, pencil sketch" ]

gen(models, samplers, prompts, base_seed, num_imgs)
