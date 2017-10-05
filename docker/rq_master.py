#!/usr/bin/env python

"""
    rq_master.py
"""

from __future__ import print_function

import os
import sys

if sys.version_info.major < 3:
    raise Exception('sys.version_info.major < 3')
    os._exit(1)
    
import time
import atexit
import argparse
from rq import Queue
from redis import Redis
from tqdm import tqdm
from collections import deque
from datetime import datetime

from rq_functions import RQFunctions, QNASControllers

# --
# Args

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--initial-jobs', type=int, default=1)
    parser.add_argument('--controller', type=str, default='DummyController')
    
    parser.add_argument('--empty', action='store_true')
    parser.add_argument('--result-ttl', type=int, default=60 * 60 * 6) # 6 hours
    parser.add_argument('--ttl', type=int, default=60 * 60 * 6) # 6 hours
    parser.add_argument('--timeout', type=int, default=60 * 10) # 5 minutes
    parser.add_argument('--keep-workers', action="store_true")
    
    return parser.parse_args()


class RQMaster(object):
    
    def __init__(self, **kwargs):
        
        self.jobs = deque()
        
        connection = Redis(
            host=kwargs['redis_host'],
            port=kwargs['redis_port'],
            password=kwargs['redis_password']
        )
        
        self.job_queue = Queue('jobs', connection=connection)
        self.kill_queue = Queue('kill', connection=connection)
        self.empty()
        
        self.ttl = kwargs['ttl']
        self.result_ttl = kwargs['result_ttl']
        self.timeout = kwargs['timeout']
        self.n_failures = 0
        self.n_success = 0
    
    def empty(self):
        print("RQMaster.empty: called", file=sys.stderr)
        _ = self.job_queue.empty()
        _ = self.kill_queue.empty()
    
    def add_job(self, jobspec, sleep_interval=None):
        func = RQFunctions.get(jobspec['func'])
        del jobspec['func']
        
        jobspec.update({
            "ttl" : self.ttl,
            "result_ttl" : self.result_ttl,
            "timeout" : self.timeout,
        })
        self.jobs.append(self.job_queue.enqueue(func, **jobspec))
        if sleep_interval:
            time.sleep(sleep_interval)
    
    def run_loop(self, success_callback, failure_callback):
        counter = 0
        while True:
            if len(self.jobs) == 0:
                return
            
            job = self.jobs.popleft()
            
            if job.is_finished:
                self.n_success += 1
                print('n_success=%d' % self.n_success, file=sys.stderr)
                next_jobspec = success_callback(job.result)
                if next_jobspec:
                    self.add_job(next_jobspec)
            elif job.is_failed:
                self.n_failures += 1
                print('n_failures=%d' % self.n_failures, file=sys.stderr)
                next_jobspec = failure_callback(job.result)
                if next_jobspec:
                    self.add_job(next_jobspec)
            else:
                self.jobs.append(job)
            
            counter += 1
            if not counter % 100:
                time.sleep(1)
                print("RQMaster.run_loop: %d jobs | %s" %
                        (len(self.jobs), datetime.now().strftime('%Y-%m-%d %H:%M:%S')), file=sys.stderr)
    
    def kill_workers(self, n_workers=100):
        print("RQMaster.kill_workers: called", file=sys.stderr)
        for _ in range(n_workers):
            _ = self.kill_queue.enqueue(RQFunctions.get('kill'), ttl=self.ttl, timeout=60 * 60 * 12)



if __name__ == "__main__":
    
    try:
        credentials = {
            'redis_host'     : os.environ['QNAS_HOST'],
            'redis_port'     : os.environ['QNAS_PORT'],
            'redis_password' : os.environ['QNAS_PASSWORD'],
        }
    except:
        raise Exception("couldn't get credentials -- try `source credentials.sh`")
    
    args = parse_args()
    kwargs = vars(args)
    kwargs.update(credentials)
    
    # Start RQMaster
    master = RQMaster(**kwargs)
    
    if args.empty:
        _ = master.empty()
        os._exit(0)
    
    # Kill workers
    if not args.keep_workers:
        atexit.register(master.kill_workers)
    
    # Initialize controller
    controller = QNASControllers[args.controller]()
    
    # Add initial jobs
    for _ in range(args.initial_jobs):
        master.add_job(controller.seed())
    
    # Run, possibly adding more jobs
    master.run_loop(controller.success, controller.failure)
