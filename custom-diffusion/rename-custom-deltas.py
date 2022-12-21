#!/usr/bin/python

from pathlib import Path
import sys
import os

def system(cmd: str):
    print(cmd)
    os.system(cmd)

def remove_diffusers_model(modeldir: Path):
    to_remove = []
    for d in ["feature_extractor", "logs", "model_index.json", "safety_checker", "scheduler", "text_encoder", "tokenizer", "unet", "vae"]:
        path = Path(modeldir, d)
        if path.exists():
            system(f"rm -r {path}")

def go_modeldir(modeldir: Path):
    modeldir = modeldir.absolute()
    for subdir in modeldir.iterdir():
        if not subdir.is_dir() or not subdir.name.startswith("checkpoint-"):
            continue

        delta_src = Path(subdir, "delta.bin")
        if delta_src.exists():
            steps = int(subdir.name.replace("checkpoint-", ""))
            delta_dest = Path(modeldir, f"delta-{modeldir.name}_{steps}.bin")
            cmd = f"mv {delta_src} {delta_dest}"
            system(cmd)

        remove_diffusers_model(subdir)
        system(f"rmdir {subdir}")

    txt_src = Path(modeldir, "train-custom-diffusion.txt")
    if txt_src.exists():
        txt_dest = Path(modeldir, f"train-custom-diffusion--{modeldir.name}.txt")
        cmd = f"mv {txt_src} {txt_dest}"
        system(cmd)
    
    remove_diffusers_model(modeldir)


for arg in sys.argv[1:]:
    go_modeldir(Path(arg))

