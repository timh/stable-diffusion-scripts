#!/bin/bash

if [[ $HUGGINGFACE_TOKEN ]]
then
    umask 0700
    mkdir -p ~/.huggingface
    echo "$HUGGINGFACE_TOKEN" > ~/.huggingface/token
    git config --global credential.helper store
fi

if [[ $PUBLIC_KEY ]]
then
    umask 0700
    mkdir -p ~/.ssh
    echo $PUBLIC_KEY >> ~/.ssh/authorized_keys

    cd /
    service ssh start
    echo "SSH Service Started"
fi

sleep infinity
