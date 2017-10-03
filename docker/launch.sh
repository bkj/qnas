#!/bin/bash

# launch.sh
#
# Run worker in docker container

source credentials.sh

# --
# Run a GPU worker
nvidia-docker run \
    -e QNAS_HOST=$QNAS_HOST \
    -e QNAS_PORT=$QNAS_PORT \
    -e QNAS_PASSWORD=$QNAS_PASSWORD \
    qnas /root/projects/qnas/docker/run-worker.sh

# --
# .. or run lots of workers on CPUs ..

# for i in $(seq 12); do
#     sudo docker run -d \
#         -e QNAS_HOST=$QNAS_HOST \
#         -e QNAS_PORT=$QNAS_PORT \
#         -e QNAS_PASSWORD=$QNAS_PASSWORD \
#         qnas /root/projects/qnas/docker/run-worker.sh
# done