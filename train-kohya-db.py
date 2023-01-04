from pathlib import Path
import math
import subprocess
from typing import Tuple

IMAGE_DIR = "/home/tim/devel/images.alex/kohya-10"
REG_DIR = "/home/tim/devel/class_images-MirageML/kohya-Mix"
LR_SCHED = "constant"
SEED = 0
BATCH = 10

def count_images(path: Path) -> int:
    count = 0
    for subpath in path.iterdir():
        if subpath.suffix in [".jpg", ".jpeg", ".png"]:
            count += 1
        elif subpath.is_dir():
            count += count_images(subpath)
    return count

NUM_IMAGES = count_images(Path(IMAGE_DIR))

STEPS_PER_EPOCH = math.ceil(NUM_IMAGES * 2 / BATCH)
NUM_EPOCHS = 250
STEPS = STEPS_PER_EPOCH * NUM_EPOCHS
SAVE_EPOCHS = 50

BASE = "runwayml/stable-diffusion-v1-5"
BASE_SHORT = "sd15"

TEXT_LR = "2.0"
LR = "1.0"

def find_resume_dir(outdir: Path, num_epochs: int) -> Tuple[Path, int]:
    resume_subdir: Path = None
    max_epoch: int = 0
    if Path(outdir).exists():
        for subdir in Path(outdir).iterdir():
            print(f"find_resume_dir: subdir {subdir}")
            if subdir.name.startswith("epoch-") and subdir.name.endswith("-state"):
                name = subdir.name.replace("epoch-", "").replace("-state", "")
                epoch = int(name)
                print(f"  epoch = {epoch}, num_epochs {num_epochs}, max_epoch {max_epoch}")
                if epoch > num_epochs:
                    continue
                if epoch > max_epoch:
                    resume_subdir = subdir
                max_epoch = max(epoch, max_epoch)
    print(f"  resume_subdir {resume_subdir}, max_epoch {max_epoch}")
    return (resume_subdir, max_epoch)

if __name__ == "__main__":
    for TEXT_EPOCH in [50, 100, 150]:
        TEXT_STEPS = TEXT_EPOCH * STEPS_PER_EPOCH
        OUTDIR = [f"alex{NUM_IMAGES}",
                    "kohya",
                    BASE_SHORT,
                    f"batch{BATCH}",
                    f"te{TEXT_STEPS:03}_{TEXT_LR}"]
        OUTDIR = "-".join(OUTDIR)
        OUTDIR = f"/home/tim/models/{OUTDIR}@{LR}_r{SEED}"

        args = [
            "accelerate", "launch", "--num_cpu_threads_per_process=4", "train_db.py",
            f"--pretrained_model_name_or_path={BASE}",
            f"--output_dir={OUTDIR}",
            f"--train_data_dir={IMAGE_DIR}",
            f"--reg_data_dir={REG_DIR}",
            f"--lr_scheduler={LR_SCHED}",
            "--lr_warmup_steps=0",
            f"--train_batch_size={BATCH}",
            f"--seed={SEED}",
            "--prior_loss_weight=1.0",
            "--gradient_checkpointing", "--use_8bit_adam", "--xformers",
            "--mixed_precision=bf16",
            f"--save_every_n_epochs={SAVE_EPOCHS}",
            "--save_state",
            f"--logging_dir={OUTDIR}/logs",
            f"--stop_text_encoder_training={TEXT_STEPS}",
            "--resolution=512,512"
        ]

        Path(OUTDIR).mkdir(exist_ok=True)
        filename = Path(OUTDIR, "train-kohya.sh")

        # text
        text_args = list(args)
        text_args.append(f"--learning_rate={TEXT_LR}e-6")
        text_args.append(f"--max_train_steps={TEXT_STEPS}")
        resume_subdir, resume_epoch = find_resume_dir(OUTDIR, TEXT_STEPS)
        print(f"resume_subdir {resume_subdir}, resume_epoch {resume_epoch}")
        if resume_subdir is not None:
            text_args.extend(["--resume", str(resume_subdir)])
        
        text_argstr = "\n  ".join(text_args)
        with open(filename, "w") as file:
            print(f"TEXT_STEPS = {TEXT_STEPS} @ {TEXT_LR}e-6", file=file)
            print(f"     STEPS = {STEPS} @ {LR}e-6", file=file)
            print("# text", file=file)
            print(text_argstr, file=file)
        
        print(f"\nTEXT_ARGS: {text_argstr}")
        if resume_epoch == int(TEXT_STEPS / STEPS_PER_EPOCH):
            print(" .. not running, nothing to do")
        else:
            subprocess.run(text_args, check=True)

        # unet
        unet_args = list(args)
        unet_args.append(f"--learning_rate={LR}e-6")
        unet_args.append(f"--max_train_steps={STEPS}")
        resume_subdir, resume_epoch = find_resume_dir(OUTDIR, STEPS)
        if resume_subdir is not None:
            unet_args.extend(["--resume", str(resume_subdir)])

        unet_argstr = "\n  ".join(unet_args)
        with open(filename, "a") as file:
            print("# unet", file=file)
            print(unet_argstr, file=file)

        print(f"\nUNET_ARGS: {unet_argstr}")
        if resume_epoch == int(STEPS / STEPS_PER_EPOCH):
            print(" .. not running, nothing to do")
        else:
            subprocess.run(unet_args, check=True)
            pass
