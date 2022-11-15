import os, os.path
import argparse
import subprocess
from typing import List

# for seed in 666 777
# do (
# export OUTPUT_DIR="$TOP/outputs/alexhin24-inpainting5_${seed}"
# accelerate launch train_inpainting_dreambooth.py \
#   --output_dir=$OUTPUT_DIR \
#   --class_data_dir=$CLASS_DIR \
#   --instance_data_dir=$INSTANCE_DIR \
#   --instance_prompt="photo of alexhin" \
#   --class_prompt="photo of a white woman" \
#   --num_class_images=76 \
#   --learning_rate=2e-6 \
#   --save_interval=1000 \
#   --save_infer_steps=35 \
# \
#   --seed=$seed \
#   --pretrained_model_name_or_path=$MODEL_NAME  \
#   --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
#   --with_prior_preservation --prior_loss_weight=1.0 \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --train_text_encoder \
#   --sample_batch_size=1 \
#   --gradient_accumulation_steps=1 --gradient_checkpointing \
#   --use_8bit_adam \
#   --lr_scheduler="polynomial" \
#   --lr_warmup_steps=0 \
#   --max_train_steps=$STEPS \
#   --mixed_precision=fp16 \
#   --not_cache_latents \
#   --hflip
# )
# done

class Config(argparse.Namespace):
    input_model_name: str

    def validate(self):
        if not os.path.isdir(self.class_dir):
            raise Exception(f"class_dir {self.class_dir} doesn't exist")
        if not os.path.isdir(self.instance_dir):
            raise Exception(f"instance_dir {self.instance_dir} doesn't exist")

        self.output_dir = f"{self.output_root}/{self.name}"
        if self.inpainting:
            self.input_model_name = "runwayml/stable-diffusion-inpainting"
        else:
            self.input_model_name = "runwayml/stable-diffusion-v1-5"
        
    def _output_dir(self, seed:int):
        return f"{self.output_root}/{self.name}_r{seed}"

    def _find_prior_steps_trained(self, output_dir) -> int:
        if not os.path.exists(output_dir):
            return 0
        
        max_steps: int = 0
        for dirname in os.listdir(output_dir):
            if not all([c.isdigit() for c in dirname]):
                continue
            max_steps = max(max_steps, int(dirname))
        
        return max_steps
    
    def run_one(self, seed):
        input_model_name = self.input_model_name

        output_dir_base = self._output_dir(seed)
        prior_steps = self._find_prior_steps_trained(output_dir_base)
        if prior_steps != 0:
            print(f"{output_dir_base}: found existing model with {prior_steps} steps already trained")
            input_model_name = f"{output_dir_base}/{prior_steps}"
            max_train_steps = self.max_train_steps - prior_steps
            output_dir = f"{output_dir_base}+{prior_steps}"
        else:
            output_dir = output_dir_base
            max_train_steps = self.max_train_steps
        
        if max_train_steps <= 0:
            print(f"** nothing to do, max_train_steps is {max_train_steps}")
            return

        train_py = "train_inpainting_dreambooth.py" if self.inpainting else "train_dreambooth.py"
        args = ["accelerate", "launch", train_py,
                "--output_dir", output_dir,
                "--instance_data_dir", self.instance_dir,
                "--instance_prompt", self.instance_prompt,
                "--class_prompt", self.class_prompt,
                "--learning_rate", str(self.learning_rate),
                "--save_interval", str(self.save_interval),
                "--seed", str(seed),
                "--pretrained_model_name_or_path", input_model_name,
                "--pretrained_vae_name_or_path=stabilityai/sd-vae-ft-mse",
                "--with_prior_preservation",
                "--prior_loss_weight=1.0",
                "--resolution=512",
                "--train_batch_size=1",
                "--train_text_encoder",
                "--sample_batch_size=1",
                "--gradient_accumulation_steps=1",
                "--gradient_checkpointing",
                "--use_8bit_adam",
                "--lr_scheduler=constant",
                "--lr_warmup_steps=0",
                "--max_train_steps", str(max_train_steps),
                "--mixed_precision=fp16"]

        if self.inpainting:
            args.extend(["--save_infer_steps=35", 
                         "--not_cache_latents",
                         "--hflip"])

        # write the config we used for this         
        for steps in range(self.save_interval, max_train_steps+1, self.save_interval):
            dirname = f"{output_dir}/{steps}"
            filename = f"{dirname}/train-cmdline.txt"
            os.makedirs(dirname, exist_ok=True)
            with open(filename, "w") as output:
                output.write(f"{' '.join(args)}\n")
                output.write(f"# --max_train_steps: {self.max_train_steps}\n")
                output.write(f"# --output_root: {self.output_root}\n")
        
        print(f"run_one:")
        print(f"       output_dir: {output_dir}")
        print(f"  max_train_steps: {self.max_train_steps}")
        print(f"     instance_dir: {self.instance_dir}")
        print(f"   instance_prompt: {self.instance_prompt}")
        print(f"         class_dir: {self.class_dir}")
        print(f"      class_prompt: {self.class_prompt}")
        print(f"              args: {' '.join(args)}")

        # res = subprocess.run(args, capture_output=True, check=True)
        # print(res.stdout)

        if prior_steps != 0:
            for steps in range(self.save_interval, max_train_steps+1, self.save_interval):
                src = f"{output_dir}/{steps}"
                dest = f"{output_dir_base}/{steps + prior_steps}"
                args = ["mv", src, dest]

                print(f"run_one: prior steps: moving {src} to {dest}")
                res = subprocess.run(args, capture_output=True, check=True)
                if res.stdout:
                    print(str(res.capture_output, "utf-8"))
    def run(self):
        for seed in self.seeds:
            self.run_one(seed)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="training wrapper for dreambooth")
    parser.add_argument("--output_root", type=str, default="/workspace/outputs", help="Path to root of output directory")
    parser.add_argument("--name", type=str, required=True, help="name of model to train, e.g., alexhin20")
    parser.add_argument("--class_dir", type=str, default="/workspace/class.white-woman", help="class training images directory")
    parser.add_argument("--class_prompt", type=str, default="photo of white woman", help="class prompt")
    parser.add_argument("--instance_dir", type=str, default="/workspace/images.alex-24", help="instance images directory")
    parser.add_argument("--instance_prompt", type=str, default="photo of alexhin", help="instance prompt")
    parser.add_argument("--learning_rate", type=str, default="2e-6", help="learning rate")
    parser.add_argument("--seeds", type=str, default="1", help="random seeds (comma separated for multiple)")
    parser.add_argument("--max_train_steps", type=int, required=True, default=2000, help="number of training steps")
    parser.add_argument("--inpainting", type=bool, help="start with inpainting instead of normal model")
    parser.add_argument("--save_interval", type=int, default=1000, help="save every <N> steps")

    cfg = Config()
    args = parser.parse_args(None, namespace=cfg)

    cfg.validate()

    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    cfg.run()
