#!/bin/bash

# launch.sh
#
# Run worker in docker container

# --
# Run a GPU worker

source credentials.sh
sudo NV_GPU=$1 nvidia-docker run --shm-size=1g -it \
    -e QNAS_HOST=$QNAS_HOST \
    -e QNAS_PORT=$QNAS_PORT \
    -e QNAS_PASSWORD=$QNAS_PASSWORD \
    qnas3 /root/projects/qnas/docker/run-worker.sh
