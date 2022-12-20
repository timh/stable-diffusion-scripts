import os, os.path
import sys
import argparse
from pathlib import Path
import subprocess
import shlex
import datetime
import hashlib
from typing import List

def args_to_quoted_str(args: List[str]) -> str:
    def one_arg(s: str) -> str:
        if " " in s:
            return '"' + s + '"'
        return s
    
    return " ".join(map(one_arg, args))

class Config(argparse.Namespace):
    input_model_name: str

    def validate(self):
        if not self.noclass:
            if self.class_dir is None or self.class_prompt is None:
                raise Exception("must pass class_dir and class_prompt, or use --noclass")
            
            if not os.path.isdir(self.class_dir):
                raise Exception(f"class_dir {self.class_dir} doesn't exist")
        elif self.class_dir is not None or self.class_prompt is not None:
            raise Exception(f"passed --noclass, but class_dir {self.class_dir} or class_prompt {self.class_prompt}" is set)

        if not os.path.isdir(self.instance_dir):
            raise Exception(f"instance_dir {self.instance_dir} doesn't exist")

    def _output_dir(self, seed:int) -> str:
        return f"{self.output_root}/{self.name}_r{seed}"

    def _find_prior_steps_trained(self, output_dir:str) -> int:
        if not os.path.exists(output_dir):
            return 0
        
        max_steps: int = 0
        for dirname in os.listdir(output_dir):
            if not dirname.startswith("checkpoint-"):
                continue
            steps = int(dirname.split("-")[-1])
            max_steps = max(max_steps, steps)
        
        return max_steps
    
    def run_one(self, seed:int):
        input_model_name = self.input_model_name
        output_dir_base = self._output_dir(seed)
        output_dir = output_dir_base

        max_train_steps = 0
        prior_steps = 0
        if self.max_train_steps:
            prior_steps = self._find_prior_steps_trained(output_dir_base)
            max_train_steps = self.max_train_steps - prior_steps
            if prior_steps != 0:
                print(f"{output_dir_base}: found existing model with {prior_steps} steps already trained")
                input_model_name = f"{output_dir_base}/checkpoint-{prior_steps}"
                output_dir = f"{output_dir_base}+{prior_steps}"
        
            if max_train_steps <= 0:
                print(f"** nothing to do, max_train_steps is {max_train_steps}")
                return
        
        use_shivam = True # True if we're using github.com/ShivamShrirao/diffusers
        
        class_args: List[str] = []
        num_class_images = 0
        if self.class_prompt is not None:
            class_args.extend(["--class_prompt", self.class_prompt])
            class_args.extend(["--class_data_dir", self.class_dir])
            num_class_images = len(list(Path(self.class_dir).iterdir()))
            class_args.extend(["--num_class_images", str(num_class_images)])
            class_args.append("--with_prior_preservation")
            class_args.append("--prior_loss_weight=1.0")

        dreambooth_py: str = "train_inpainting_dreambooth.py" if "inpainting" in self.input_model_name else "train_dreambooth.py"
        args = ["accelerate", "launch",
                "--num_cpu_threads_per_process", "4",
                dreambooth_py,
                "--output_dir", output_dir,
                "--instance_data_dir", self.instance_dir,
                "--instance_prompt", self.instance_prompt,
                *class_args,
                "--learning_rate", str(self.learning_rate),
                "--lr_scheduler", self.lr_scheduler,
                "--train_batch_size", str(self.train_batch_size),
                # "--save_interval", str(self.save_interval),
                # "--save_min_steps", str(self.save_min_steps),
                # "--save_infer_steps=50",
                # "--n_save_sample=10",
                "--seed", str(seed),
                "--pretrained_model_name_or_path", input_model_name,
                # "--pretrained_vae_name_or_path=stabilityai/sd-vae-ft-mse",
                # "--resolution=512",
                "--train_text_encoder",
                "--sample_batch_size=1",
                "--gradient_accumulation_steps=1",
                "--gradient_checkpointing",
                "--use_8bit_adam",
                "--lr_warmup_steps=0",
                "--mixed_precision=bf16",
                # "--checkpointing_steps=10000"
                ]

        if self.save_interval and use_shivam:
            args.extend(["--save_interval", str(self.save_interval)])
            args.extend(["--save_min_steps", str(int(max_train_steps / 2))])
        elif self.save_interval:
            args.extend(["--save_steps", str(self.save_interval)])

        if self.save_epochs:
            args.extend(["--save_epochs", str(self.save_epochs)])
        if max_train_steps:
            args.extend(["--max_train_steps", str(max_train_steps)])
        if self.num_train_epochs:
            args.extend(["--num_train_epochs", str(self.num_train_epochs)])

        if use_shivam and "inpainting" in self.input_model_name:
            args.append("--not_cache_latents")

        print(f"run_one:")
        print(f"     output_dir: {output_dir}")
        print(f"max_train_steps: {self.max_train_steps}")
        print(f"   instance_dir: {self.instance_dir}")
        print(f"instance_prompt: {self.instance_prompt}")
        print(f"      class_dir: {self.class_dir}")
        print(f"   class_prompt: {self.class_prompt}")
        print(f"           args: {args_to_quoted_str(args)}")

        txt_filename = Path(output_dir).joinpath("train-cmdline.txt")
        if not self.dry_run:
            print(f"** write {txt_filename.absolute()}")
            txt_filename.parent.mkdir(exist_ok=True)
            with open(txt_filename, "w") as output:
                output.write(f"{dreambooth_py}:\n")
                output.write(f"# {args_to_quoted_str(args)}\n")
                output.write(f"\n")
                output.write(f"train.py:\n")
                output.write(f"# {args_to_quoted_str(sys.argv[1:])}\n")

                paths = [path for path in Path(self.instance_dir).iterdir() if path.suffix in [".jpg", ".jpeg", ".png", ".webp"]]

                output.write("\n")
                output.write(f"instance_dir {self.instance_dir} has {len(paths)} images:\n")
                for image_path in paths:
                    hashstr = hashlib.sha256(open(image_path, "rb").read()).hexdigest()
                    output.write(f"  {image_path.name}: sha256 {hashstr}\n")

            # place the config in the global training history
            timestr = datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d-%H%M%S.txt")
            global_path = Path(self.output_root, "training-history", timestr)
            global_path.parent.mkdir(exist_ok=True)
            print(f"** write {global_path.absolute()}")
            copy_args = ["cp", txt_filename.absolute(), global_path.absolute()]
            subprocess.run(copy_args)

            res = subprocess.run(args, stdout=sys.stdout, stderr=sys.stderr, check=True)
            print(res.stdout)

            # copy the config to all the checkpoint directories
            for subdir in Path(output_dir).iterdir():
                if not subdir.is_dir() or not subdir.name.startswith("checkpoint-"):
                    continue
                copy_args = ["cp", txt_filename.absolute(), subdir.absolute()]
                print(f"** write {subdir.absolute()}/train-cmdline.txt")
                res = subprocess.run(copy_args)

        if prior_steps != 0:
            for steps in range(self.save_interval, max_train_steps+1, self.save_interval):
                src = f"{output_dir}/checkpoint-{steps}"
                dest = f"{output_dir_base}/checkpoint-{steps + prior_steps}"
                move_args = ["mv", src, dest]

                print(f"run_one: prior steps: moving {src} to {dest}")
                if not self.dry_run:
                    res = subprocess.run(move_args, stdout=sys.stdout, stderr=sys.stderr, check=True)
                    if res.stdout:
                        print(str(res.capture_output, "utf-8"))
    def run(self):
        for seed in self.seeds.split(","):
            self.run_one(seed)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="training wrapper for dreambooth", fromfile_prefix_chars="@")
    parser.add_argument("--output_root", default="/workspace/outputs", help="Path to root of output directory")
    parser.add_argument("--name", "-n", required=True, help="name of model to train, e.g., alexhin20")
    parser.add_argument("--class_dir", help="class training images directory")
    parser.add_argument("--class_prompt", help="class prompt")
    parser.add_argument("--noclass", action='store_true', help="must pass this if no class dir/prompt is included")
    parser.add_argument("--instance_dir", required=True, help="instance images directory")
    parser.add_argument("--instance_prompt", required=True, help="instance prompt")
    parser.add_argument("--learning_rate", "--lr", "-l", default="2e-6", help="learning rate")
    parser.add_argument("--lr_scheduler", default="polynomial", help="scheduler type: constant, linear, cosine, polynomial")
    parser.add_argument("--seeds", "-S", default="1", help="random seeds (comma separated for multiple)")
    parser.add_argument("--steps", "-s", dest='max_train_steps', type=int, help="number of training steps")
    parser.add_argument("--epochs", dest='num_train_epochs', type=int, help="number epochs")
    parser.add_argument("--model", dest='input_model_name', default="runwayml/stable-diffusion-v1-5", help="name or path for base model")
    parser.add_argument("--save_interval", type=int, default=500, help="save every <N> steps")
    parser.add_argument("--save_epochs", type=int, default=0, help="save every <N> epochs")
    parser.add_argument("--save_min_steps", type=int, default=500, help="only save checkpoints at or greater than <N> steps")
    parser.add_argument("--train_batch_size", type=int, default=1, help="train batch size")
    parser.add_argument("--dry_run", default=False, help="dry run: don't do actions", action='store_true')

    cfg = Config()

    # optional ~/.sdscripts.conf file including default arguments
    config_filename = os.path.join(os.environ["HOME"], ".sdscripts.conf")
    if os.path.exists(config_filename):
        with open(config_filename, "r") as config:
            config_args = shlex.split(config.read(), comments=True)
        print(f"read {args_to_quoted_str(config_args)}")
        config_args.extend(sys.argv[1:])
    else:
        config_args = sys.argv[1:]
    
    parser.parse_args(config_args, namespace=cfg)
    cfg.validate()

    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    cfg.run()
