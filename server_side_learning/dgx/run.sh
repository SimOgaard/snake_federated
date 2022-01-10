#!/bin/bash
set -e

IMAGE_NAME="snake_federated"

ROOT_DIR=$PWD

username=$(whoami)

docker run -it --rm --runtime nvidia \
-e CUDA_VISIBLE_DEVICES=3 \
# ALT 2 (comment if you want to do ALT 1)
-v $ROOT_DIR:/home/$username \
$IMAGE_NAME \

python3 snake_federated_transfer_learning.py