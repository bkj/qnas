#!/bin/bash

# launch.sh
#
# Run worker in docker container

source credentials.sh

nvidia-docker run \
    -e QNAS_HOST=$QNAS_HOST \
    -e QNAS_PORT=$QNAS_PORT \
    -e QNAS_PASSWORD=$QNAS_PASSWORD \
    qnas /root/projects/qnas/docker/run-worker.sh
