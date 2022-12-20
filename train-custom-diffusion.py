import shlex
from pathlib import Path
import subprocess

MODEL_NAME = "runwayml/stable-diffusion-v1-5"
# MODEL_NAME = "/home/tim/models/stable-diffusion-v1-5+vae"

# BATCH = 2
BATCH = 1
MODIFIER_TOKEN = None
TRAIN_TEXT_ENCODER = True
FREEZE_CROSSATTN = True

LR = "5.0e-6"
INSTANCE_PROMPT = "timhin person"
INSTANCE_DIR = "/home/tim/devel/images.tim/15-512px"
CLASS_DIR = "/home/tim/devel/class_images-MirageML/Mix"
CLASS_PROMPT = "person"
NUM_INSTANCES = len(list(Path(INSTANCE_DIR).iterdir()))
STEPS = NUM_INSTANCES * 500
SAVE_STEPS = int(STEPS / 10)

extra = ""
if TRAIN_TEXT_ENCODER:
    extra = extra + "+text"
if BATCH > 1:
    extra = extra + f"+batch{BATCH}"
if FREEZE_CROSSATTN:
    extra = extra + "+crossattn"

OUTPUT_DIR = f"/home/tim/models/tim15-custom/timperson15{extra}@{LR}"

args = [
    "accelerate", "launch", "src/diffuser_training.py",

    "--pretrained_model_name_or_path", MODEL_NAME,
    "--output_dir", OUTPUT_DIR,

    "--instance_data_dir", INSTANCE_DIR,
    "--instance_prompt", INSTANCE_PROMPT,
    *(["--modifier_token", MODIFIER_TOKEN] if MODIFIER_TOKEN else []),
    *(["--train_text_encoder"] if TRAIN_TEXT_ENCODER else []),
    *(["--freeze_model", "crossattn"] if FREEZE_CROSSATTN else []),

    "--class_data_dir", CLASS_DIR,
    "--class_prompt", CLASS_PROMPT,
    "--with_prior_preservation", "--real_prior", "--prior_loss_weight=1.0",
    "--num_class_images=200",
    
    "--resolution=512",
    "--train_batch_size", str(BATCH),
    "--learning_rate", LR,
    "--lr_warmup_steps=0",
    "--max_train_steps", str(STEPS),
    "--save_steps", str(SAVE_STEPS),
    "--scale_lr",

    "--mixed_precision", "no", # must be set to 'no'. if 'bf16' or 'fp16', it crashes due to unimplemented function
    *(["--gradient_checkpointing", "--use_8bit_adam"] if BATCH > 1 else []),
]

cmdline = shlex.join(args)
print(cmdline)
txtfile = Path(OUTPUT_DIR, "train-custom-diffusion.txt")
txtfile.parent.mkdir(exist_ok=True)
with open(txtfile, "w") as txt:
    txt.write(cmdline + "\n")

subprocess.run(args)
