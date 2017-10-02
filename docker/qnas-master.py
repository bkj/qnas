#!/usr/bin/env python

"""
    qnas-master.py
"""

import sys
import time
from rq import Queue
from redis import Redis
from collections import deque

from qnas_funcs import *

result_ttl = 60 * 60 * 2 # 2 hours
ttl = 60 * 60 * 2 # 2 hours

# --
# Create queue

results = deque()
q = Queue(connection=Redis())
q.empty()

# --
# Enqueue jobs

def initialize():
    jobs = range(2)
    
    for job in jobs:
        time.sleep(0.01)
        print >> sys.stderr, 'enqueuing %s' % str(job)
        results.append(q.enqueue(run_job, {'a' : job}, ttl=ttl, result_ttl=result_ttl))
    
    return results

# --
# Wait for jobs to finish

def run(results, callback=None):
    while True:
        if len(results) == 0:
            break
        
        r = results.popleft()
        
        if r.is_finished:
            params, result = r.result
            
            if callback is not None:
                callback(params, result)
            
        elif r.is_failed:
            print >> sys.stderr, 'failed'
        else:
            results.append(r)


def callback(params, result):
    # results.append(q.enqueue(run_job, result, ttl=ttl, result_ttl=result_ttl))
    print >> sys.stderr, 'done: %s' % str(params)
    print 'n_remaining=%d' % len(results)


if __name__ == "__main__":
    results = initialize()
    run(results, callback)