#!/bin/bash

# qnas-worker.sh
#
# Start worker

cd /root/projects/qnas/docker/
git pull

rqworker --url redis://:$QNAS_PASSWORD@$QNAS_HOST:$QNAS_PORT 2>&1 | tee -a log