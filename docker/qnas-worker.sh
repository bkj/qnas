#!/bin/bash

# qnas-worker.sh
#
# Start worker

echo $REDIS_URL
rqworker --url 10.105.0.3 2>&1 | tee -a log