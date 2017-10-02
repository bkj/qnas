#!/bin/bash

# qnas-worker.sh
#
# Start worker

URL="10.105.0.3"
rqworker --url $URL 2>&1 | tee -a log