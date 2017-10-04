#!/bin/bash

# run-master.sh

source credentials.sh

# Run jobs
python master.py \
    --redis-password $QNAS_PASSWORD \
    --redis-host $QNAS_HOST \
    --redis-port $QNAS_PORT

# Clean up queue
python master.py \
    --redis-password $QNAS_PASSWORD \
    --redis-host $QNAS_HOST \
    --redis-port $QNAS_PORT \
    --empty


