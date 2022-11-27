FROM continuumio/miniconda3:latest
#FROM ubuntu:jammy-20221101

RUN apt-get update && \
    apt-get install -y zsh tmux emacs-nox vim curl git-core openssh-server rsync git-lfs zip less locales && \
    /opt/conda/bin/conda update -n base -c defaults conda
#RUN apt-get install -y nvidia-kernel-common-510-server nvidia-utils-510-server nvidia-dkms-510-server nvidia-compute-utils-510-server
#RUN apt-get install -y nvidia-utils-510-server nvidia-compute-utils-510-server
#RUN apt-get install -y python3-pip python3-pycuda python3.10-venv

#####
# diffusers
##### 
ENV PATH=/opt/conda/envs/diffusers/bin:$PATH
RUN git clone https://github.com/ShivamShrirao/diffusers /workspace/diffusers && \
    conda create -n diffusers && \
    conda install pip
    
ENV CONDA_DEFAULT_ENV=diffusers
WORKDIR /workspace/diffusers
#RUN --mount=type=cache,target=/opt/conda/envs,id=conda_diffusers \
RUN conda install pip pytorch torchvision pytorch-cuda=11.6 transformers -c pytorch -c nvidia && \
    pip install accelerate && \
    pip install -e .


# # class images for dreambooth training
# WORKDIR /workspace
# RUN git clone https://github.com/JoePenna/Stable-Diffusion-Regularization-Images class_images

# # RUN pip install jupyterlab

# # WORKDIR /workspace
# # RUN git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui
# # WORKDIR /workspace/stable-diffusion-webui
# # RUN pip install -r requirements.txt


# WORKDIR /workspace
# RUN python3 -mvenv /venvs/diffusers
# RUN /venvs/diffusers/bin/pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
# RUN /venvs/diffusers/bin/pip install git+https://github.com/ShivamShrirao/diffusers.git
# RUN /venvs/diffusers/bin/pip install -r examples/dreambooth/requirements.txt
# RUN /venvs/diffusers/bin/pip install bitsandbytes

# WORKDIR /workspace
# RUN git clone https://github.com/invoke-ai/InvokeAI
# WORKDIR /workspace/InvokeAI
# RUN apt-get install -y libgl-dev
# RUN cp environments-and-requirements/requirements-lin-cuda.txt requirements.txt
# RUN python3 -mvenv /venvs/invokeai
# RUN /venvs/invokeai/bin/pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
# RUN /venvs/invokeai/bin/pip install "test-tube>=0.7.5"
# RUN /venvs/invokeai/bin/pip install -r requirements.txt

RUN chsh -s /bin/zsh root && \
    bash -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" && \
    git lfs install && \
    mkdir -p /workspace/outputs && \
    git clone https://github.com/timh/stable-diffusion-scripts /workspace/scripts && \
    ln -s /workspace/scripts /scripts

# the default theme 'robbyrussell' messes up in redraw with long command lines
# and tmux. change to a multiline one.
COPY tmux.conf /root/.tmux.conf
COPY zshrc /root/.zshrc
COPY cache--huggingface--accelerate--default_config.yml /root/.cache/huggingface/accelerate/default_config.yaml
RUN ln -sf /usr/share/zoneinfo/US/Pacific /etc/localtime && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    mkdir -p /root/.cache/huggingface/accelerate/

EXPOSE 22

COPY start.sh /start.sh
CMD bash /start.sh
