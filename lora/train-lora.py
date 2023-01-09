from pathlib import Path
import shlex
import subprocess
import argparse

LR_SCHEDULER = "linear"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr_unet", default="1.0e-4")
    parser.add_argument("--lr_text", default="1.0e-5")
    parser.add_argument("--lr_ti", default="5.0e-4")
    parser.add_argument("--steps_ti", type=int, default=1000)
    parser.add_argument("--steps_tuning", type=int, default=1000)
    parser.add_argument("-n", "--name", type=str, required=True)
    parser.add_argument("--instance_dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tokens", type=str, default="<alex>")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--face", dest="use_face_segmentation_condition", action="store_true", default=False)

    cfg = parser.parse_args()

    cfg.model_short = Path(cfg.model).name
    model_short = {
        "stable-diffusion-v1-5" : "sd15",
        "stable-diffusion-2-1" : "sd21",
        "stable-diffusion-2-1-base" : "sd21base",
        "stable-diffusion-2-0" : "sd20",
        "stable-diffusion-2-0-base" : "sd20base"
    }.get(cfg.model_short)
    if model_short is not None:
        cfg.model_short = model_short

    num_images = len(list(Path(cfg.instance_dir).iterdir()))
    name_args = [
        f"{cfg.name}{num_images:02}",
        cfg.model_short,
        f"batch{cfg.batch}",
        *(["faceseg"] if cfg.use_face_segmentation_condition else []),
    ]
    weight_args = [
        f"ti_{cfg.steps_ti}@{cfg.lr_ti}",
        f"unet_{cfg.steps_tuning}@{cfg.lr_unet}",
        f"text_{cfg.steps_tuning}@{cfg.lr_text}",
    ]
    name = "-".join(name_args) + "--" + ",".join(weight_args)
    output_dir = f"/home/tim/models/{name}"

    args = [
        # "lora_pti",
        "accelerate", "launch", "/home/tim/micromamba/envs/lora/bin/lora_pti",

        "--output_dir", output_dir,
        "--pretrained_model_name_or_path", cfg.model,

        "--instance_data_dir", cfg.instance_dir,

        "--max_train_steps_ti", str(cfg.steps_ti),
        "--max_train_steps_tuning", str(cfg.steps_tuning),
        "--save_steps=100",
        "--learning_rate_unet", cfg.lr_unet,
        "--learning_rate_text", cfg.lr_text,
        "--learning_rate_ti", cfg.lr_ti,
        "--lr_scheduler", LR_SCHEDULER,
        "--lr_warmup_steps=0",
        "--train_text_encoder",

        "--perform_inversion=True",
        "--clip_ti_decay",
        "--weight_decay_ti=0.000",
        "--weight_decay_lora=0.001",
        "--continue_inversion",
        "--continue_inversion_lr=1e-4",
        "--lora_rank=1",

        "--placeholder_tokens", cfg.tokens,
        "--initializer_tokens=person",
        "--use_template=object",
        *(["--use_face_segmentation_condition"] if cfg.use_face_segmentation_condition else []),

        "--resolution=512",
        "--train_batch_size", str(cfg.batch),
        "--use_8bit_adam",
        "--gradient_accumulation_steps=1",
        "--color_jitter",
        "--device=cuda:0",

        "--mixed_precision", "bf16"
    ]


    cmdline = shlex.join(args)
    print(cmdline)
    txtfile = Path(output_dir, "train-lora.txt")
    txtfile.parent.mkdir(exist_ok=True)
    with open(txtfile, "w") as txt:
        txt.write(cmdline)
    
    subprocess.run(args)
