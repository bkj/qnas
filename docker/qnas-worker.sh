#!/bin/bash

# qnas-worker.sh
#
# Start worker

cd /root/projects/qnas/docker/
rqworker --url $REDIS_URL 2>&1 | tee -a log