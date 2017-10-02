#!/bin/bash

# qnas-worker.sh
#
# Start worker

cd /root/projects/qnas/docker/
git pull
rqworker --url $REDIS_URL--burst 2>&1 | tee -a log