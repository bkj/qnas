#!/usr/bin/env python

"""
    deploy-ec2.py
    
    Deploy QNAS3 Docker workers via EC2
    
    !! This references my security groups, SSH key and private AMI
    However, should be easy to use
    
        ./provision-qnas-ec2.sh
        
    to build your own AMI
"""

import os
import sys
import boto3
import base64
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance-count', type=int, default=1)
    parser.add_argument('--image-id', type=str, default='ami-a770b5dd')
    parser.add_argument('--gpu', action="store_true")
    parser.add_argument('--njobs', type=int, default=1)
    
    args = parser.parse_args()
    
    if args.gpu:
        args.instance_type = 'p2.xlarge'
        args.spot_price = 0.25
    else:
        args.instance_type = 'm3.large'
        args.spot_price = 0.03
    
    return args

# --
# Helpers

def make_cpu_cmd(credentials, njobs=1):
    cmd = b"""#!/bin/bash

for i in $(seq %d); do
    sudo docker run -d \
        -e QNAS_HOST=%s \
        -e QNAS_PORT=%s \
        -e QNAS_PASSWORD=%s \
        qnas3 /root/projects/qnas/docker/run-worker.sh
done

while true; do
    if [ $(sudo docker ps | wc -l) -eq 1 ]; then
        echo "no docker containers -- halting!"
        sudo halt
    fi
    sleep 5
done

    """ % (
        njobs,
        credentials['QNAS_HOST'],
        credentials['QNAS_PORT'],
        credentials['QNAS_PASSWORD']
    )
    
    return cmd, base64.b64encode(cmd)


def make_gpu_cmd(credentials, njobs=1):
    cmd = b"""#!/bin/bash

sudo nvidia-smi -pm ENABLED -i 0
sudo nvidia-smi -ac 2505,875 -i 0

for i in $(seq %d); do
    sudo nvidia-docker run -d \
        -e QNAS_HOST=%s \
        -e QNAS_PORT=%s \
        -e QNAS_PASSWORD=%s \
        qnas3 /root/projects/qnas/docker/run-worker.sh
done

while true; do
    if [ $(sudo docker ps | wc -l) -eq 1 ]; then
        echo "no docker containers -- halting!"
        sudo halt
    fi
    sleep 5
done

    """ % (
        njobs,
        credentials['QNAS_HOST'],
        credentials['QNAS_PORT'],
        credentials['QNAS_PASSWORD']
    )
    
    return cmd, base64.b64encode(cmd)


def launch_spot(client, cmd_b64, instance_count, image_id, instance_type, spot_price):
    return client.request_spot_instances(
        DryRun=False,
        SpotPrice=str(spot_price),
        InstanceCount=instance_count,
        Type='one-time',
        LaunchSpecification={
            'ImageId'        : image_id, # qnas-docker-v0
            'KeyName'        : 'rzj',
            'SecurityGroups' : ['ssh'],
            'InstanceType'   : instance_type,
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
    
    try:
        credentials = {
            'QNAS_HOST'     : os.environ['QNAS_HOST'],
            'QNAS_PORT'     : os.environ['QNAS_PORT'],
            'QNAS_PASSWORD' : os.environ['QNAS_PASSWORD'],
        }
    except:
        raise Exception("couldn't get credentials -- try `source ../credentials.sh`")
    
    make_cmd = make_gpu_cmd if args.gpu else make_cpu_cmd
    cmd_clear, cmd_b64 = make_cmd(credentials, args.njobs)
    print cmd_clear
    
    print launch_spot(
        client,
        cmd_b64,
        args.instance_count,
        args.image_id,
        args.instance_type,
        args.spot_price
    )

