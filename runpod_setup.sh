set -e

cat >~/.tmux.conf <<EOF
unbind C-b
set-option -g prefix C-o
bind-key C-o send-prefix
EOF

# system packages..
apt update
apt install -y zsh tmux emacs-nox vim
git clone https://github.com/JoePenna/Stable-Diffusion-Regularization-Images class_images

# 
cd /workspace
git clone https://github.com/ShivamShrirao/diffusers/

# python stuff
source /venv/bin/activate
pip install bitsandbytes

cd /workspace/diffusers/examples/dreambooth
pip install -U -r requirements.txt
accelerate config


# hugging face login.
huggingface-cli login
git config --global credential.helper store

# lastly, install oh-my-zsh
sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
