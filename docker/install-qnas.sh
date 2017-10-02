#!/bin/bash

mkdir -p ~/projects
cd ~/projects

git clone https://github.com/bkj/qnas
cd qnas

# Download + format datasets
./utils/download.sh
