#!/usr/bin/env python

"""
    controller/base.py
"""

import os
import sys
import time
import json
from rq import Queue
from redis import Redis
from tqdm import tqdm
from collections import deque

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
                self.fail_counter += 1
                print >> sys.stderr, 'fail_counter=%d' % self.fail_counter
            else:
                self.jobs.append(job)
    
    def kill_workers(self, delay=1, n_workers=100):
        # !! Should add check that workers are actually dead
        for _ in range(n_workers):
            _ = self.q.enqueue(kill, ttl=self.ttl, timeout=self.timeout)

# --
# Example

class DummyController(BaseController):
    
    def initialize(self, n_jobs=2):
        for i in tqdm(range(n_jobs)):
            self.enqueue(run_dummy, {
                "config" : {"model_name" : "test-%d" % i},
                "net_class" : 'mnist_net',
                "dataset" : 'MNIST',
                "cuda" : False,
                "epochs" : 1
            })
    
    def callback(self, result):
        config, hist = result
        print >> sys.stderr, 'job finished: %s' % json.dumps(config)
    
