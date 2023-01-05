from pathlib import Path
import math
import subprocess
from typing import Tuple
import argparse

def count_images(path: Path) -> int:
    count = 0
    for subpath in path.iterdir():
        if subpath.suffix in [".jpg", ".jpeg", ".png"]:
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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="training wrapper for kohya repo dreambooth", fromfile_prefix_chars="@")
    parser.add_argument("--output_root", default="/home/tim/models", help="Path to root of output directory")
    parser.add_argument("--name", "-n", required=True, help="base name of model to train, e.g., alex")
    parser.add_argument("--reg_dir", default="/home/tim/devel/class_images-MirageML/kohya-Mix", help="regularization images directory")
    parser.add_argument("--instance_dir", required=True)
    parser.add_argument("--epochs", dest="num_epochs", type=int, default=300, help="total epochs to train")
    parser.add_argument("--text_epochs", dest="num_text_epochs_list", action='append', nargs="+", help="total text epochs to train (can be multiple)")
    parser.add_argument("--lr", default="1.0e-6", help="learning rate for the steps after text training")
    parser.add_argument("--text_lr", default="2.0e-6", help="learning rate for the steps after text training")
    parser.add_argument("--model", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--save_epochs", type=int, default=-1)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)

    cfg = parser.parse_args()

    # swizzle arguments.
    cfg.num_images = count_images(Path(cfg.instance_dir))
    cfg.steps_per_epoch = math.ceil(cfg.num_images * 2 / cfg.batch)
    
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
    
    if cfg.num_text_epochs_list is None:
        cfg.num_text_epochs_list = [["25", "50", "100", "150"]]
    
    cfg.num_text_epochs_list = map(int, [int(inside) for outside in cfg.num_text_epochs_list for inside in outside])

    cfg.lr_short = cfg.lr.replace("e-6", "")
    cfg.text_lr_short = cfg.text_lr.replace("e-6", "")

    return cfg

if __name__ == "__main__":
    cfg = parse_args()

    for num_text_epochs in cfg.num_text_epochs_list:
        num_text_steps = num_text_epochs * cfg.steps_per_epoch
        outdir = [f"{cfg.name}{cfg.num_images}",
                    "kohya",
                    cfg.model_short,
                    f"batch{cfg.batch}",
                    f"text_epochs{num_text_epochs:03}",
                    f"text_steps{num_text_steps:04}_{cfg.text_lr_short}"]
        outdir = "-".join(outdir)
        outdir = f"{cfg.output_root}/{outdir}@{cfg.lr_short}_r{cfg.seed}"

        result_text = f"{outdir}/epoch-{num_text_epochs:06}"
        result_unet = f"{outdir}/epoch-{cfg.num_epochs:06}"

        print(f"result_text {result_text}")
        print(f"result_unet {result_unet}")
        if Path(result_unet).exists():
            print(f"skipping {result_unet}, already completed")
            continue

        args = [
            "accelerate", "launch", "--num_cpu_threads_per_process=4", "train_db.py",
            f"--pretrained_model_name_or_path={cfg.model}",
            f"--output_dir={outdir}",
            f"--train_data_dir={cfg.instance_dir}",
            f"--reg_data_dir={cfg.reg_dir}",
            f"--lr_scheduler=constant",
            "--lr_warmup_steps=0",
            f"--train_batch_size={cfg.batch}",
            f"--seed={cfg.seed}",
            "--prior_loss_weight=1.0",
            "--gradient_checkpointing", "--use_8bit_adam", "--xformers",
            "--mixed_precision=bf16",
            f"--save_every_n_epochs={cfg.save_epochs}",
            "--save_state",
            f"--logging_dir={outdir}/logs",
            f"--stop_text_encoder_training={num_text_steps}",
            "--resolution=512,512"
        ]
        # import sys; sys.exit(0)

        Path(outdir).mkdir(exist_ok=True)
        filename = Path(outdir, "train-kohya.txt")

        # text
        text_args = list(args)
        text_args.append(f"--learning_rate={cfg.text_lr}")
        text_args.append(f"--max_train_steps={num_text_steps}")
        resume_subdir, resume_epoch = find_resume_dir(outdir, num_text_epochs)
        print(f"for text, resume_subdir {resume_subdir}, resume_epoch {resume_epoch}")
        if resume_subdir is not None:
            text_args.extend(["--resume", str(resume_subdir)])
        
        text_argstr = "\n  ".join(text_args)
        with open(filename, "w") as file:
            print(f"TEXT_STEPS = {num_text_steps} @ {cfg.text_lr}", file=file)
            print(f"     STEPS = {cfg.num_steps} @ {cfg.lr}", file=file)
            print("# text", file=file)
            print(text_argstr, file=file)
        
        print(f"\nTEXT_ARGS: {text_argstr}")
        print(f"   resume_epoch {resume_epoch}")
        print(f"num_text_epochs {num_text_epochs}")

        if Path(result_text).exists():
            print(" .. not running, nothing to do")
        else:
            subprocess.run(text_args, check=True)

        # unet
        unet_args = list(args)
        unet_args.append(f"--learning_rate={cfg.lr}")
        unet_args.append(f"--max_train_steps={cfg.num_steps}")
        resume_subdir, resume_epoch = find_resume_dir(outdir, num_text_steps)
        print(f"for unet, resume_subdir {resume_subdir}, resume_epoch {resume_epoch}")
        if resume_subdir is not None:
            unet_args.extend(["--resume", str(resume_subdir)])

        unet_argstr = "\n  ".join(unet_args)
        with open(filename, "a") as file:
            print("# unet", file=file)
            print(unet_argstr, file=file)

        print(f"\nUNET_ARGS: {unet_argstr}")
        if resume_epoch == cfg.num_epochs:
            print(" .. not running, nothing to do")
        else:
            subprocess.run(unet_args, check=True)

