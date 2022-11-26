FROM ubuntu:jammy-20221101

COPY tmux.conf /root/.tmux.conf

RUN apt update
RUN apt install -y zsh tmux emacs-nox vim curl git-core openssh-server rsync git-lfs zip
RUN apt install -y python3-pip python3-pycuda python3.10-venv
RUN chsh -s /bin/zsh root
RUN bash -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
RUN ln -s /usr/share/zoneinfo/US/Pacific /etc/localtime
RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen
RUN echo "export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu" >> ~/.zshrc

# the default theme 'robbyrussell' messes up in redraw with long command lines
# and tmux. change to a multiline one.
RUN sed -i -e s/robbyrussell/pmcgee/g ~/.zshrc

RUN git lfs install

RUN mkdir -p /workspace/outputs
RUN mkdir -p /workspace/models

# RUN curl https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh > /usr/anaconda.bash
# https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh
COPY anaconda.bash /tmp
RUN (echo ""; echo "yes"; echo "/opt/anaconda3"; echo "yes") | bash /tmp/anaconda.bash
RUN rm /tmp/anaconda.bash

# RUN pip install jupyterlab

# RUN mkdir /venvs

# RUN mkdir -p /root/.cache/huggingface/accelerate/
# COPY cache--huggingface--accelerate--default_config.yml /root/.cache/huggingface/accelerate/default_config.yaml

# WORKDIR /workspace
# RUN git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui
# WORKDIR /workspace/stable-diffusion-webui
# RUN pip install -r requirements.txt

# WORKDIR /workspace
# RUN git clone https://github.com/ShivamShrirao/diffusers
# WORKDIR /workspace/diffusers
# RUN python3 -mvenv /venvs/diffusers
# RUN /venvs/diffusers/bin/pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
# RUN /venvs/diffusers/bin/pip install git+https://github.com/ShivamShrirao/diffusers.git
# RUN /venvs/diffusers/bin/pip install -r examples/dreambooth/requirements.txt
# RUN /venvs/diffusers/bin/pip install bitsandbytes

# WORKDIR /workspace
# RUN git clone https://github.com/invoke-ai/InvokeAI
# WORKDIR /workspace/InvokeAI
# RUN apt install -y libgl-dev
# RUN cp environments-and-requirements/requirements-lin-cuda.txt requirements.txt
# RUN python3 -mvenv /venvs/invokeai
# RUN /venvs/invokeai/bin/pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
# RUN /venvs/invokeai/bin/pip install "test-tube>=0.7.5"
# RUN /venvs/invokeai/bin/pip install -r requirements.txt

# WORKDIR /workspace
# RUN git clone https://github.com/timh/stable-diffusion-scripts scripts
# RUN ln -s /workspace/scripts /scripts

# WORKDIR /workspace
# RUN git clone https://github.com/JoePenna/Stable-Diffusion-Regularization-Images class_images

EXPOSE 22

COPY start.sh /start.sh
CMD bash /start.sh
