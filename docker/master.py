#!/usr/bin/env python

"""
    master.py
"""

import os
import sys
import time
from rq import Queue
from redis import Redis
from tqdm import tqdm
from collections import deque

from worker import *
sys.path.append('..')

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
# Controller classes

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
    
    def kill_workers(self, n_workers=100):
        # !! Should add check that workers are actually dead
        for _ in range(n_workers):
            _ = self.q.enqueue(kill, ttl=self.ttl, timeout=self.timeout)


class SimpleController(BaseController):
    
    def callback(self, result):
        """ 
            action to take when results are returned 
            ... eg, most interesting things are going to enqueue more jobs ...
        """
        config, hist = result
        print >> sys.stderr, 'job finished: %s' % json.dumps(config)
        
        # result_path = os.path.join('results', args.run, config['model_name'])
        # if not os.path.exists(os.path.dirname(result_path)):
        #     os.makedirs(os.path.dirname(result_path))
        
        # open(result_path, 'w').write('\n'.join(map(json.dumps, result)))


if __name__ == "__main__":
    
    args = parse_args()
    controller = SimpleController(args)
    
    if args.empty:
        _ = controller.q.empty()
        os._exit(0)
    
    n_jobs = 100
    for i in tqdm(range(n_jobs)):
        controller.enqueue(run_dummy, {
            "config" : {"model_name" : "test-%d" % i},
            "net_class" : 'mnist_net',
            "dataset" : 'MNIST',
            "cuda" : False,
            "epochs" : 1
        })
    
    controller.run_loop()
    controller.kill_workers()