import os, os.path
import sys
import argparse
import subprocess
import shlex
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
            if not all([c.isdigit() for c in dirname]):
                continue
            max_steps = max(max_steps, int(dirname))
        
        return max_steps
    
    def run_one(self, seed:int):
        input_model_name:str = self.input_model_name

        output_dir_base = self._output_dir(seed)
        prior_steps = self._find_prior_steps_trained(output_dir_base)
        max_train_steps:int = self.max_train_steps - prior_steps
        output_dir = output_dir_base
        if prior_steps != 0:
            print(f"{output_dir_base}: found existing model with {prior_steps} steps already trained")
            input_model_name = f"{output_dir_base}/{prior_steps}"
            output_dir = f"{output_dir_base}+{prior_steps}"
        
        if max_train_steps <= 0:
            print(f"** nothing to do, max_train_steps is {max_train_steps}")
            return

        dreambooth_py: str = "train_inpainting_dreambooth.py" if "inpainting" in self.input_model_name else "train_dreambooth.py"
        args = ["accelerate", "launch",
                "--num_cpu_threads_per_process", "4",
                dreambooth_py,
                "--output_dir", output_dir,
                "--instance_data_dir", self.instance_dir,
                "--instance_prompt", self.instance_prompt,
                "--learning_rate", str(self.learning_rate),
                "--save_interval", str(self.save_interval),
                "--save_min_steps", str(self.save_min_steps),
                "--save_infer_steps=50",
                "--seed", str(seed),
                "--pretrained_model_name_or_path", input_model_name,
                "--pretrained_vae_name_or_path=stabilityai/sd-vae-ft-mse",
                "--resolution=512",
                "--train_batch_size", str(self.train_batch_size),
                "--train_text_encoder",
                "--sample_batch_size=1",
                "--n_save_sample=10",
                "--gradient_accumulation_steps=2",
                "--gradient_checkpointing",
                "--use_8bit_adam",
                "--lr_scheduler", self.lr_scheduler,
                "--lr_warmup_steps=0",
                "--max_train_steps", str(max_train_steps)]

        if self.class_prompt is not None:
            args.extend(["--class_prompt", self.class_prompt])
            args.extend(["--class_data_dir", self.class_dir])
            args.append("--with_prior_preservation")
            args.append("--prior_loss_weight=1.0")

        if "inpainting" in self.input_model_name:
            args.append("--not_cache_latents")
        else:
            args.extend(["--save_sample_prompt", self.instance_prompt])


        print(f"run_one:")
        print(f"       output_dir: {output_dir}")
        print(f"  max_train_steps: {self.max_train_steps}")
        print(f"     instance_dir: {self.instance_dir}")
        print(f"   instance_prompt: {self.instance_prompt}")
        print(f"         class_dir: {self.class_dir}")
        print(f"      class_prompt: {self.class_prompt}")
        print(f"              args: {args_to_quoted_str(args)}")

        if not self.dry_run:
            res = subprocess.run(args, stdout=sys.stdout, stderr=sys.stderr, check=True)
            print(res.stdout)

            # write the config we used for this
            for steps in range(self.save_min_steps, max_train_steps+1, self.save_interval):
                dirname = f"{output_dir}/{steps}"
                filename = f"{dirname}/train-cmdline.txt"
                os.makedirs(dirname, exist_ok=True)
                with open(filename, "w") as output:
                    output.write(f"{dreambooth_py}:\n")
                    output.write(f"# {args_to_quoted_str(args)}\n")
                    output.write(f"\n")
                    output.write(f"train.py:\n")
                    output.write(f"# {args_to_quoted_str(sys.argv[1:])}\n")
        
        if prior_steps != 0:
            for steps in range(self.save_interval, max_train_steps+1, self.save_interval):
                src = f"{output_dir}/{steps}"
                dest = f"{output_dir_base}/{steps + prior_steps}"
                args = ["mv", src, dest]

                print(f"run_one: prior steps: moving {src} to {dest}")
                if not self.dry_run:
                    res = subprocess.run(args, stdout=sys.stdout, stderr=sys.stderr, check=True)
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
    parser.add_argument("--learning_rate", "--lr", "-l", default="1e-6", help="learning rate")
    parser.add_argument("--lr_scheduler", default="cosine", help="scheduler type: constant, linear, cosine")
    parser.add_argument("--seeds", "-S", default="1", help="random seeds (comma separated for multiple)")
    parser.add_argument("--steps", "-s", dest='max_train_steps', type=int, required=True, default=2000, help="number of training steps")
    parser.add_argument("--model", dest='input_model_name', default="runwayml/stable-diffusion-v1-5", help="name or path for base model")
    parser.add_argument("--save_interval", type=int, default=500, help="save every <N> steps")
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
