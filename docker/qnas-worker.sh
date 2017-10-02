#!/bin/bash

# qnas-worker.sh
#
# Start worker

URL='redis://10.105.0.3:6379/1'
rqworker --url $URL 2>&1 | tee -a log