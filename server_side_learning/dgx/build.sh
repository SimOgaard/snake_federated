#!/bin/bash
set -e

IMAGE_NAME="snake_federated"

docker build \
--no-cache \
--build-arg u_id=$(id -u) \
--build-arg g_id=$(id -g) \
--build-arg username=$(id -gn $USER)  \
-f Dockerfile -t $IMAGE_NAME .
