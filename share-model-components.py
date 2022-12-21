#!/usr/bin/python3

from pathlib import Path
from typing import Dict, List
import hashlib

MODEL_DIR = Path("/home/tim/models")
BASE_MODELS = ["stable-diffusion-v1-5", "stable-diffusion-v1-5+vae", "f222", "f222v", "hassanblend1.4"]

# key = sha, value = path relative to /home/tim/models.
HASH: Dict[str, Path] = dict()

OUT = open("share-model.bash", "w")
SIZE_SAVED = 0

def walk_special_dir(special_dir: Path, record_shas: bool):
    global SIZE_SAVED

    output = ""
    all_same = True
    last_existing_parent = None
    symlinks = []

    for child in special_dir.iterdir():
        if child.name == ".git":
            continue

        if child.is_dir():
            raise Exception(f"unexpected child {child} in {special_dir.absolute()}")
        elif child.is_symlink() or child.name.endswith(".json") or child.name.endswith(".sha256"):
            continue

        # cache the sha256 we compute for next time.
        sha_file = Path(child.with_suffix(child.suffix + ".sha256"))
        if sha_file.exists():
            sha = open(sha_file, "r").read()
        else:
            sha = hashlib.sha256(open(child, "rb").read()).hexdigest()
            open(sha_file, "w").write(sha)

        filename = child.relative_to(MODEL_DIR)

        if sha in HASH:
            existing_filename = HASH[sha]
            if last_existing_parent is not None and last_existing_parent != existing_filename.parent:
                all_same = False
            last_existing_parent = existing_filename.parent
            output += f"\033[1m{sha}  |  {filename} == {existing_filename}\033[0m\n"
            symlinks.append([filename, existing_filename])
            SIZE_SAVED += Path(MODEL_DIR, filename).stat().st_size
        else:
            all_same = False
            if record_shas:
                HASH[sha] = filename

    if all_same and last_existing_parent is not None:
        special_path = special_dir.relative_to(MODEL_DIR)
        #last_existing_parent = last_existing_parent.absolute().relative_to(MODEL_DIR)
        print(f"\033[1m{'<various>':32}  |  {special_path} == {last_existing_parent}\033[0m")
        print(f"rm -fr {special_path}", file=OUT)
        print(f"ln -s {last_existing_parent.absolute()} {special_path}", file=OUT)
    else:
        print(output, end="")
        for src, dest in symlinks:
            print(f"ln -sf {dest.absolute()} {src}", file=OUT)

def walk_model_dir(model_dir: Path, record_shas: bool):
    for subdir in model_dir.iterdir():
        if subdir.is_symlink():
            # some subdirs are already symlinked. don't report duplicates on them.
            continue
        
        if subdir.name in ["feature_extractor", "safety_checker", "scheduler", "text_encoder", "tokenizer", "unet", "vae"]:
            walk_special_dir(subdir, record_shas)
        elif subdir.name.startswith("checkpoint-") or subdir.name.startswith("save-"):
            walk_model_dir(subdir, record_shas)
        elif subdir.name in ["model_index.json", "logs"] or subdir.suffix in [".pt", ".bin", ".txt", ".pkl"]:
            pass
        else:
            print(f"ignore {subdir}")
            pass

def walk_models():
    for model_dir in MODEL_DIR.iterdir():
        if not model_dir.is_dir():
            continue
        if model_dir.name in BASE_MODELS or model_dir.name == "sd-vae-ft-mse":
            continue
        walk_model_dir(model_dir, False)

if __name__ == "__main__":
    walk_special_dir(MODEL_DIR / "sd-vae-ft-mse", True)
    for name in BASE_MODELS:
        base_model_path = MODEL_DIR / name
        walk_model_dir(base_model_path, True)
    
    print(f"size saved so far: {SIZE_SAVED} bytes, {SIZE_SAVED/1024/1024} Mb, {SIZE_SAVED/1024/1024/1024} Gb")
    walk_models()
    
    print(f"size saved at end: {SIZE_SAVED} bytes, {SIZE_SAVED/1024/1024} Mb, {SIZE_SAVED/1024/1024/1024} Gb")

