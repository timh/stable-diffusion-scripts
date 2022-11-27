#!/bin/bash

if [[ $HUGGINGFACE_TOKEN ]]
then
    umask 0077
    mkdir -p ~/.huggingface
    echo -n "$HUGGINGFACE_TOKEN" > ~/.huggingface/token
    git config --global credential.helper store
fi

if [[ $PUBLIC_KEY ]]
then
    umask 0077
    mkdir -p ~/.ssh
    echo $PUBLIC_KEY >> ~/.ssh/authorized_keys

    cd /
    service ssh start
    echo "SSH Service Started"
fi

conda activate diffusers
jupyter-lab --allow-root --notebook-dir=/ >& /var/log/jupyter.log &
tensorboard --logdir /workspace/outputs >& /var/log/tensorboard.log &

sleep infinity
