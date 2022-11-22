#!/usr/bin/env python3

import os, os.path

OUTPUTS_DIR = "/workspace/outputs"
SAMPLES_DIR = "/workspace/output-samples"

if __name__ == "__main__":
    # generate directories & filenames of the form that imagegrid looks for..
    for model_name in os.listdir(OUTPUTS_DIR):
        model_dir = f"{OUTPUTS_DIR}/{model_name}"
        dir0 = f"{model_dir}/0"
        if not os.path.isdir(dir0):
            continue

        for steps in os.listdir(model_dir):
            steps_dir = f"{model_dir}/{steps}"

            if not os.path.isdir(steps_dir) or steps == "0":
                continue
            if not all(lambda c: c.isdigit() for c in steps):
                continue
            
            link_dir = f"{SAMPLES_DIR}/{model_name}_{int(steps):04}-sample-ddim_50"
            os.makedirs(link_dir, exist_ok=True)

            samples_dir = f"{steps_dir}/samples"
            for sample in os.listdir(samples_dir):
                print(f"sample {sample}")
                if not sample.endswith(".png"):
                    continue

                sample_filename = f"{samples_dir}/{sample}"
                link_filename = f"{link_dir}/00.{sample}"
                
                os.system(f"ln -sfv {sample_filename} {link_filename}")
    
    # then call imagegrid on the output so we can look at them all together.
    os.chdir(SAMPLES_DIR)
    os.system("python /scripts/imagegrid.py > imagegrid.html")

