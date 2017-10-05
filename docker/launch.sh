#!/bin/bash

# launch.sh
#
# Run worker in docker container

# --
# Run a GPU worker

source credentials.sh
sudo nvidia-docker run -it \
    -e QNAS_HOST=$QNAS_HOST \
    -e QNAS_PORT=$QNAS_PORT \
    -e QNAS_PASSWORD=$QNAS_PASSWORD \
    qnas3 /root/projects/qnas/docker/run-worker.sh

# --
# .. or run lots of workers on CPUs ..

source credentials.sh
for i in $(seq 12); do
    sudo docker run -d \
        -e QNAS_HOST=$QNAS_HOST \
        -e QNAS_PORT=$QNAS_PORT \
        -e QNAS_PASSWORD=$QNAS_PASSWORD \
        qnas3 /root/projects/qnas/docker/run-worker.sh
done