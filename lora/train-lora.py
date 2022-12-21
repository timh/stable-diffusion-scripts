from pathlib import Path
import shlex
import subprocess

BATCH = 1
LR_SCHEDULER = "linear"

# MODEL_NAME="/home/tim/models/f222v"
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
# CLASS_DIR = "/home/tim/devel/class_images/person_ddim"
CLASS_DIR = "/home/tim/devel/class_images-MirageML/Mix"
CLASS_PROMPT = "photo of person"
use_classes = True

if False:
    INSTANCE_DIR = "/home/tim/devel/images.alex/20-from30-512px"
    INSTANCE_PROMPT = "photo of alexhin"
    BASENAME = "alex20-sd15"
else:
    INSTANCE_DIR = "/home/tim/devel/images.tim/15-512px"
    INSTANCE_PROMPT = "photo of timhin"
    BASENAME = "tim15-sd15"

BASENAME = f"{BASENAME}-{LR_SCHEDULER}"
if use_classes:
    BASENAME += "-withpp"
else:
    BASENAME += "-nopp"

num_images = len(list(Path(INSTANCE_DIR).iterdir()))
STEPS = num_images * 200


for LR in ["1e-4"]:
    for LR_TEXT in ["5e-5"]:
        OUTPUT_DIR = f"/home/tim/models/{BASENAME}-batch{BATCH}@{LR},{LR_TEXT}_{STEPS}"

        class_args = []
        if use_classes:
            class_args = ["--class_data_dir", CLASS_DIR,
                          "--class_prompt", CLASS_PROMPT,
                          "--with_prior_preservation",
                          "--prior_loss_weight=1.0"]


        args = [
            "accelerate", "launch", "train_lora_dreambooth.py",

            "--output_dir", OUTPUT_DIR,
            "--pretrained_model_name_or_path", MODEL_NAME,

            "--instance_data_dir", INSTANCE_DIR,
            "--instance_prompt", INSTANCE_PROMPT,

            *class_args,

            "--max_train_steps", str(STEPS),
            "--learning_rate", LR,
            "--learning_rate_text", LR_TEXT,
            "--lr_scheduler", LR_SCHEDULER,
            "--lr_warmup_steps=0",
            "--train_text_encoder",

            "--resolution=512",
            "--train_batch_size", str(BATCH),
            "--use_8bit_adam",
            "--gradient_accumulation_steps=1",
            "--color_jitter",

            "--mixed_precision", "bf16"
        ]

        cmdline = shlex.join(args)
        print(cmdline)
        txtfile = Path(OUTPUT_DIR, "train-lora.txt")
        txtfile.parent.mkdir(exist_ok=True)
        with open(txtfile, "w") as txt:
            txt.write(cmdline)
        
        subprocess.run(args)


# master $ python train_lora_dreambooth.py -h                                                                                      [17:57:21]

#                                 [--adam_epsilon ADAM_EPSILON] [--max_grad_norm MAX_GRAD_NORM] [--push_to_hub] [--hub_token HUB_TOKEN]
#                                 [--logging_dir LOGGING_DIR] [--mixed_precision {no,fp16,bf16}] [--local_rank LOCAL_RANK]
#                                 [--resume_unet RESUME_UNET] [--resume_text_encoder RESUME_TEXT_ENCODER]

# Simple example of a training script.

# options:
#   -h, --help            show this help message and exit
#   --pretrained_model_name_or_path PRETRAINED_MODEL_NAME_OR_PATH
#                         Path to pretrained model or model identifier from huggingface.co/models.
#   --pretrained_vae_name_or_path PRETRAINED_VAE_NAME_OR_PATH
#                         Path to pretrained vae or vae identifier from huggingface.co/models.
#   --revision REVISION   Revision of pretrained model identifier from huggingface.co/models.
#   --tokenizer_name TOKENIZER_NAME
#                         Pretrained tokenizer name or path if not the same as model_name
#   --instance_data_dir INSTANCE_DATA_DIR
#                         A folder containing the training data of instance images.
#   --class_data_dir CLASS_DATA_DIR
#                         A folder containing the training data of class images.
#   --instance_prompt INSTANCE_PROMPT
#                         The prompt with identifier specifying the instance
#   --class_prompt CLASS_PROMPT
#                         The prompt to specify images in the same class as provided instance images.
#   --with_prior_preservation
#                         Flag to add prior preservation loss.
#   --prior_loss_weight PRIOR_LOSS_WEIGHT
#                         The weight of prior preservation loss.
#   --num_class_images NUM_CLASS_IMAGES
#                         Minimal class images for prior preservation loss. If not have enough images, additional images will be sampled
#                         with class_prompt.
#   --output_dir OUTPUT_DIR
#                         The output directory where the model predictions and checkpoints will be written.
#   --seed SEED           A seed for reproducible training.
#   --resolution RESOLUTION
#                         The resolution for input images, all the images in the train/validation dataset will be resized to this resolution
#   --center_crop         Whether to center crop images before resizing to resolution
#   --color_jitter        Whether to apply color jitter to images
#   --train_text_encoder  Whether to train the text encoder
#   --train_batch_size TRAIN_BATCH_SIZE
#                         Batch size (per device) for the training dataloader.
#   --sampl
#                         Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.
#   --lora_rank LORA_RANK
#                         Rank of LoRA approximation.
#   --learning_rate LEARNING_RATE
#                         Initial learning rate (after the potential warmup period) to use.
#   --learning_rate_text LEARNING_RATE_TEXT
#                         Initial learning rate for text encoder (after the potential warmup period) to use.
#   --scale_lr            Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.
#   --lr_scheduler LR_SCHEDULER
#                         The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant",
#                         "constant_with_warmup"]
#   --lr_warmup_steps LR_WARMUP_STEPS
#                         Number of steps for the warmup in the lr scheduler.
#   --use_8bit_adam       Whether or not to use 8-bit Adam from bitsandbytes.
#   --adam_beta1 ADAM_BETA1
#                         The beta1 parameter for the Adam optimizer.
#   --adam_beta2 ADAM_BETA2
#                         The beta2 parameter for the Adam optimizer.
#   --adam_weight_decay ADAM_WEIGHT_DECAY
#                         Weight decay to use.
#   --adam_epsilon ADAM_EPSILON
#                         Epsilon value for the Adam optimizer
#   --max_grad_norm MAX_GRAD_NORM
#                         Max gradient norm.
#   --push_to_hub         Whether or not to push the model to the Hub.
#   --hub_token HUB_TOKEN
#                         The token to use to push to the Model Hub.
#   --logging_dir LOGGING_DIR
#                         [TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to
#                         *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***.
#   --mixed_precision {no,fp16,bf16}
#                         Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10.and an
#                         Nvidia Ampere GPU. Default to the value of accelerate config of the current system or the flag passed with the
#                         `accelerate.launch` command. Use this argument to override the accelerate config.
#   --local_rank LOCAL_RANK
#                         For distributed training: local_rank
#   --resume_unet RESUME_UNET
#                         File path for unet lora to resume training.
#   --resume_text_encoder RESUME_TEXT_ENCODER
#                         File path for text encoder lora to resume training.

