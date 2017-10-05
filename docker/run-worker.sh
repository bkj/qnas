#!/bin/bash

# qnas-worker.sh
#
# Start worker

export LC_ALL=C.UTF-8
export LANG=C.UTF-8
rqworker --url redis://:$QNAS_PASSWORD@$QNAS_HOST:$QNAS_PORT kill jobs 2>&1 | tee -a log
