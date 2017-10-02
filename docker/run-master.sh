#!/bin/bash

# run-master.sh

python master.py \
    --redis-password $QNAS_PASSWORD \
    --redis-host $QNAS_HOST \
    --redis-port $QNAS_PORT