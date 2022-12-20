import shlex
from pathlib import Path
import subprocess

MODEL_NAME = "runwayml/stable-diffusion-v1-5"
# MODEL_NAME = "/home/tim/models/stable-diffusion-v1-5+vae"

BATCH = 1

# defaults from their example
if False:
    LR = "1e-5"
    STEPS = 250
    MODIFIER_TOKEN = "<new1>"
    INSTANCE_PROMPT = "photo of a <new1> cat"
    INSTANCE_DIR = "./data/cat"
    CLASS_DIR = "./real_reg/samples_cat/"
    CLASS_PROMPT = "cat"
    OUTPUT_DIR = "./logs/cat"
else:
    # for human faces 5e-6
    LR = "5.0e-6"
    MODIFIER_TOKEN = "timhin"
    INSTANCE_PROMPT = "timhin person"
    INSTANCE_DIR = "/home/tim/devel/images.tim/15-512px"
    CLASS_DIR = "/home/tim/devel/class_images-MirageML/Mix"
    CLASS_PROMPT = "person"
    STEPS = len(list(Path(INSTANCE_DIR).iterdir())) * 500
    OUTPUT_DIR = f"/home/tim/models/tim15-custom/timperson15@{LR}_{STEPS:05}"

args = [
    "accelerate", "launch", "src/diffuser_training.py",

    "--pretrained_model_name_or_path", MODEL_NAME,
    "--output_dir", OUTPUT_DIR,

    "--instance_data_dir", INSTANCE_DIR,
    "--instance_prompt", INSTANCE_PROMPT,
    "--modifier_token", MODIFIER_TOKEN,

    "--class_data_dir", CLASS_DIR,
    "--class_prompt", CLASS_PROMPT,
    "--with_prior_preservation", "--real_prior", "--prior_loss_weight=1.0",
    "--num_class_images=200",
    
    "--resolution=512",
    "--train_batch_size", str(BATCH),
    "--learning_rate", LR,
    "--lr_warmup_steps=0",
    "--max_train_steps", str(STEPS),
    "--scale_lr",

    "--mixed_precision", "no"
]

cmdline = shlex.join(args)
print(cmdline)
txtfile = Path(OUTPUT_DIR, "train-custom-diffusion.txt")
txtfile.parent.mkdir(exist_ok=True)
with open(txtfile, "w") as txt:
    txt.write(cmdline + "\n")

subprocess.run(args)
