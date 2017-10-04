#!/usr/bin/env python

"""
    master.py
"""

import os
import sys
import argparse

sys.path.append('../qnas')
from qnas_trainer import QNASTrainer
from controllers.base import DummyController

# --
# Args

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=str, default='qnas-0')
    parser.add_argument('--redis-password', type=str)
    parser.add_argument('--redis-host', type=str)
    parser.add_argument('--redis-port', type=int)
    
    parser.add_argument('--empty', action='store_true')
    
    parser.add_argument('--result-ttl', type=int, default=60 * 60 * 6) # 6 hours
    parser.add_argument('--ttl', type=int, default=60 * 60 * 6) # 6 hours
    parser.add_argument('--timeout', type=int, default=60 * 60 * 6) # 6 hours
    
    return parser.parse_args()

# --
# Run

if __name__ == "__main__":
    
    args = parse_args()
    controller = DummyController(args)
    
    if args.empty:
        _ = controller.q.empty()
        os._exit(0)
    
    controller.initialize()
    controller.run_loop()
    controller.kill_workers()