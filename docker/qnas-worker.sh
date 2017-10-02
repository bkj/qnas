#!/bin/bash

# qnas-worker.sh
#
# Start worker

rqworker 2>&1 | tee -a log