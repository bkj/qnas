#!/bin/bash

# qnas-worker.sh
#
# Start worker

cd /root/projects/qnas/
git pull
python setup.py clean --all install

cd /root/projects/qnas/docker
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
rqworker --url redis://:$QNAS_PASSWORD@$QNAS_HOST:$QNAS_PORT kill jobs 2>&1 | tee -a log
