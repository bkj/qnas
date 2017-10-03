#!/bin/bash

# run-master.sh

source credentials.sh

python master.py \
    --redis-password $QNAS_PASSWORD \
    --redis-host $QNAS_HOST \
    --redis-port $QNAS_PORT

