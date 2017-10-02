#!/bin/bash

# qnas-worker.sh
#
# Start worker

printenv

echo $REDIS_URL
rqworker --url $REDIS_URL 2>&1 | tee -a log