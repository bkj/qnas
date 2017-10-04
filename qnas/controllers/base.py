#!/usr/bin/env python

"""
    controller/base.py
"""

import os
import sys
import time
import json
from datetime import datetime
from rq import Queue
from redis import Redis
from tqdm import tqdm
from collections import deque

sys.path.append('..')
from qnas_trainer import qnas_trainer_run

# --
# Helpers


def kill(delay=1):
    print >> sys.stderr, '!!! Kill Signal Received -- shutting down in %ds' % delay
    time.sleep(delay)
    os.kill(os.getppid(), 9)


def run_dummy(config, **kwargs):
    time.sleep(3)
    return config, {"dummy" : True}

# --
# Controllers

class BaseController(object):
    
    def __init__(self, args):
        self.jobs = deque()
        self.q = Queue(connection=Redis(host=args.redis_host, port=args.redis_port, password=args.redis_password))
        self.q.empty()
        
        self.ttl = args.ttl
        self.result_ttl = args.result_ttl
        self.timeout = args.timeout
        self.fail_counter = 0
    
    def enqueue(self, func, obj, sleep_interval=None):
        obj.update({
            "ttl" : self.ttl,
            "result_ttl" : self.result_ttl,
            "timeout" : self.timeout,
        })
        self.jobs.append(self.q.enqueue(func, **obj))
        if sleep_interval:
            time.sleep(sleep_interval)
    
    def run_loop(self):
        while True:
            if len(self.jobs) == 0:
                return
            
            job = self.jobs.popleft()
            
            if job.is_finished:
                self.callback(job.result)
            elif job.is_failed:
                print job
                self.fail_counter += 1
                print >> sys.stderr, 'fail_counter=%d' % self.fail_counter
            else:
                self.jobs.append(job)
            
            time.sleep(0.1)
            sys.stdout.write("\r%d jobs | %s" % (len(self.jobs), datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            sys.stdout.flush()
    
    def kill_workers(self, delay=1, n_workers=100):
        # !! Should add check that workers are actually dead
        for _ in range(n_workers):
            _ = self.q.enqueue(kill, ttl=self.ttl, timeout=self.timeout)
    
    def initialize(self, n_jobs=2):
        raise NotImplementedError()
    
    def callback(self, result):
        raise NotImplementedError()

# --
# Example

class DummyController(BaseController):
    
    def initialize(self, n_jobs=2):
        for i in tqdm(range(n_jobs)):
            self.enqueue(qnas_trainer_run, {
                "config"    : {
                    "model_name"  : "mnist-%d" % i,
                    "net_class"   : 'MNISTNet',
                    "dataset"     : 'MNIST',
                    "epochs"      : 1,
                    "lr_schedule" : 'linear',
                    "lr_init"     : 0.1,
                },
                "cuda" : True,
            })
    
    def callback(self, result):
        config, hist = result
        print >> sys.stderr, 'job finished: %s' % json.dumps(config)
        print >> sys.stderr, hist[-1]['train_acc'], hist[-1]['val_acc'], hist[-1]['test_acc']
    
