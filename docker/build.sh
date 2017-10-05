#!/bin/bash

# build.sh

sudo docker build -t qnas3 --build-arg CACHEBUST=$(date +%s) .

