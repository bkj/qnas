#!/usr/bin/env python

"""
    deploy-ec2.py
    
    Deploy a bunch of Docker images on EC2
"""

import sys
import uuid
import json
import boto3
import base64
import argparse
import numpy as np
from uuid import uuid4
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-workers', type=int, required=True)
    parser.add_argument('--image-id', type=str, required=True)
    return parser.parse_args()

# --
# Helpers

def make_cpu_cmd(credentials):
    cmd = b"""#!/bin/bash

sudo docker run -d \
    -e QNAS_HOST=%s \
    -e QNAS_PORT=%s \
    -e QNAS_PASSWORD=%s \
    qnas /root/projects/qnas/docker/run-worker.sh

echo "terminating..."
sudo halt

    """ % (
        credentials['QNAS_HOST'],
        credentials['QNAS_PORT'],
        credentials['QNAS_PASSWORD']
    )
    
    return cmd, base64.b64encode(cmd)


def make_gpu_cmd(credentials):
    cmd = b"""#!/bin/bash

sudo nvidia-smi -pm ENABLED -i 0
sudo nvidia-smi -ac 2505,875 -i 0

sudo nvidia-docker run -d \
    -e QNAS_HOST=%s \
    -e QNAS_PORT=%s \
    -e QNAS_PASSWORD=%s \
    qnas /root/projects/qnas/docker/run-worker.sh

echo "terminating..."
sudo halt

    """ % (
        credentials['QNAS_HOST'],
        credentials['QNAS_PORT'],
        credentials['QNAS_PASSWORD']
    )
    
    return cmd, base64.b64encode(cmd)


def launch_spot(client, cmd_b64, image_id, spot_price=0.25):
    return client.request_spot_instances(
        DryRun=False,
        SpotPrice=str(spot_price),
        InstanceCount=1,
        Type='one-time',
        LaunchSpecification={
            'ImageId'        : image_id, # qnas-docker-v0
            'KeyName'        : 'rzj',
            'SecurityGroups' : ['ssh'],
            'InstanceType'   : 'p2.xlarge',
            'Placement' : {
                'AvailabilityZone': 'us-east-1c', # cheapest when I looked
            },
            'UserData' : cmd_b64,
        }
    )


# --
# Run

if __name__ == "__main__":
    args = parse_args()
    client = boto3.client('ec2')
    
    credentials = {
        'QNAS_HOST' : os.environ['QNAS_HOST'],
        'QNAS_PORT' : os.environ['QNAS_PORT'],
        'QNAS_PASSWORD' : os.environ['QNAS_PASSWORD'],
    }
    
    if args.gpu:
        cmd_clear, cmd_b64 = make_gpu_cmd(credentials)
    else:
        cmd_clear, cmd_b64 = make_cpu_cmd(credentials)
    
    print >> sys.stderr, cmd_clear
    for _ in range(args.n_workers):
        print launch_spot(client, cmd_b64, image_id=args.image_id)

