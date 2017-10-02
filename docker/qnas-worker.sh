#!/bin/bash

# qnas-worker.sh
#
# Start worker

echo $REDIS_URL
rqworker --url "$REDIS_URL" 2>&1 | tee -a log