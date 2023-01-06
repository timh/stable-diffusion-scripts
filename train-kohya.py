from pathlib import Path
import math
import sys
import subprocess
from typing import Tuple, Set
import argparse

SUFFIXES = [".jpg", ".jpeg", ".png", ".webp"]
def count_images(path: Path) -> int:
    count = 0
    for subpath in path.iterdir():
        if subpath.suffix in SUFFIXES:
            count += 1
        elif subpath.is_dir():
            count += count_images(subpath)
    return count

def find_resume_dir(outdir: Path, num_epochs: int) -> Tuple[Path, int]:
    resume_subdir: Path = None
    max_epoch: int = 0
    if Path(outdir).exists():
        for subdir in Path(outdir).iterdir():
            if subdir.name.startswith("epoch-") and subdir.name.endswith("-state"):
                name = subdir.name.replace("epoch-", "").replace("-state", "")
                epoch = int(name)
                if epoch > num_epochs:
                    continue
                if epoch > max_epoch:
                    resume_subdir = subdir
                max_epoch = max(epoch, max_epoch)
    return (resume_subdir, max_epoch)

def prepare_native(cfg: argparse.Namespace, outdir: str) -> str:
    image_stems: Set[str] = set()
    caption_stems: Set[str] = set()
    for item in Path(cfg.instance_dir).iterdir():
        if item.suffix in SUFFIXES:
            image_stems.add(item.stem)
        elif item.suffix == ".caption":
            caption_stems.add(item.stem)

    num_error = 0    
    for image_stem in image_stems:
        if image_stem not in caption_stems:
            num_error += 1
            print(f"ERROR: no caption for {image_stem}")
    for caption_stem in caption_stems:
        if caption_stem not in image_stems:
            print(f"WARN: no image for {caption_stem}")
    
    if num_error > 0:
        raise ValueError(f"{num_error} images missing captions")
    
    metadata_cap_filename = f"{outdir}/meta_cap.json"
    metadata_filename = f"{outdir}/meta.json"
    
    args = [
        "python", "finetune/merge_captions_to_metadata.py",
        "--full_path", cfg.instance_dir,
        metadata_cap_filename
    ]
    print(" ".join(args))
    subprocess.run(args, check=True)

    args = [
        "python", "finetune/prepare_buckets_latents.py",
        "--full_path", cfg.instance_dir,
        metadata_cap_filename, metadata_filename,
        f"{cfg.model}/vae"
    ]
    print(" ".join(args))
    subprocess.run(args, check=True)

    return metadata_filename

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="training wrapper for kohya repo dreambooth", fromfile_prefix_chars="@")
    parser.add_argument("--output_root", default="/home/tim/models", help="Path to root of output directory")
    parser.add_argument("--name", "-n", required=True, help="base name of model to train, e.g., alex")
    parser.add_argument("--reg_dir", default=None, help="regularization images directory")
    parser.add_argument("--instance_dir", required=True)
    parser.add_argument("--epochs", dest="num_epochs", type=int, default=300, help="total epochs to train")
    parser.add_argument("--lr", default="1.0e-6", help="learning rate")
    parser.add_argument("--model", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--save_epochs", type=int, default=-1)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dreambooth", "--db", dest="train_dreambooth", default=False, action='store_true', help="run dreambooth training?")
    parser.add_argument("--native", dest="train_native", default=False, action='store_true', help="run native training?")

    cfg = parser.parse_args()

    if cfg.train_dreambooth == cfg.train_native:
        parser.error("one of --dreambooth or --native must be specified")

    if cfg.train_native and cfg.reg_dir is not None:
        parser.error("must not specify regularization images for native training")

    if cfg.train_dreambooth and cfg.reg_dir is None:
        cfg.reg_dir = "/home/tim/devel/class_images-MirageML/kohya-Mix"

    # swizzle arguments.
    cfg.num_images = count_images(Path(cfg.instance_dir))
    if cfg.train_dreambooth:
        cfg.steps_per_epoch = math.ceil(cfg.num_images * 2 / cfg.batch)
    else:
        cfg.steps_per_epoch = math.ceil(cfg.num_images / cfg.batch)
    
    cfg.num_steps = cfg.num_epochs * cfg.steps_per_epoch

    print(f"num_images = {cfg.num_images}")
    print(f"steps_per_epoch = {cfg.steps_per_epoch}")
    print(f"num_epochs = {cfg.num_epochs}")
    print(f"num_images = {cfg.num_images}")

    if cfg.save_epochs == -1:
        cfg.save_epochs = int(cfg.num_epochs / 3)

    if "stable-diffusion-v1-5" in cfg.model:
        cfg.model_short = "sd15"
    elif "stable-diffusion-2-1" in cfg.model:
        cfg.model_short = "sd21"
    elif "stable-diffusion-inpainting" in cfg.model:
        cfg.model_short = "sd15inpaint"
    else:
        cfg.model_short = Path(cfg.model).name
    
    cfg.lr_short = cfg.lr.replace("e-6", "")

    return cfg

if __name__ == "__main__":
    cfg = parse_args()

    if cfg.train_native:
        name_tag = "kohyanative"
    else:
        name_tag = "kohyadb"
    outdir = [f"{cfg.name}{cfg.num_images}",
                name_tag,
                cfg.model_short,
                f"batch{cfg.batch}"]
    outdir = "-".join(outdir)
    outdir = f"{cfg.output_root}/{outdir}@{cfg.lr_short}_r{cfg.seed}"

    epoch_dir = f"{outdir}/epoch-{cfg.num_epochs:06}"

    if Path(epoch_dir).exists():
        print(f"skipping {epoch_dir}, already completed")
        sys.exit(0)

    if cfg.train_dreambooth:
        script = "train_db.py"
    else:
        script = "fine_tune.py"
    args = [
        "accelerate", "launch", "--num_cpu_threads_per_process=4", 
        script,
        "--",
        f"--pretrained_model_name_or_path={cfg.model}",
        f"--output_dir={outdir}",
        f"--train_data_dir={cfg.instance_dir}",
        "--train_text_encoder",
        f"--learning_rate={cfg.lr}",
        f"--max_train_steps={cfg.num_steps}",
        f"--lr_scheduler=constant",
        "--lr_warmup_steps=0",
        f"--train_batch_size={cfg.batch}",
        f"--seed={cfg.seed}",
        "--gradient_checkpointing", "--use_8bit_adam", "--xformers",
        "--mixed_precision=bf16",
        f"--save_every_n_epochs={cfg.save_epochs}",
        "--save_state",
        f"--logging_dir={outdir}/logs"
    ]

    if cfg.train_dreambooth:
        args.append(f"--reg_data_dir={cfg.reg_dir}")
        args.append("--prior_loss_weight=1.0")
        args.append("--resolution=512,512")
    else:
        metadata_filename = prepare_native(cfg, outdir)
        args.extend(["--in_json", metadata_filename])

    Path(outdir).mkdir(exist_ok=True)

    resume_subdir, resume_epoch = find_resume_dir(outdir, cfg.num_epochs)
    if resume_subdir is not None:
        args.extend(["--resume", str(resume_subdir)])
    
    argstr = "\n  ".join(args)
    filename = Path(outdir, "train-kohya.txt")
    with open(filename, "w") as file:
        print(f"     STEPS = {cfg.num_steps} @ {cfg.lr}", file=file)
        print(argstr, file=file)
    
    subprocess.run(args, check=True)


