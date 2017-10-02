#!/usr/bin/env python

"""
    qnas-master.py
"""

import sys
import time
from rq import Queue
from redis import Redis
from collections import deque

from qnas_funcs import run_job

# --
# Create queue

results = deque()
q = Queue(connection=Redis())
q.empty()

# --
# Enqueue jobs

jobs = range(10)

for job in jobs:
    time.sleep(0.1)
    print >> sys.stderr, 'enqueuing %s' % str(job)
    result = q.enqueue(run_job, job)
    results.append(result)

# --
# Wait for jobs to finish

n_remaining = len(jobs)

while True:
    if len(results) == 0:
        break
    
    r = results.popleft()
    
    if r.is_finished:
        params, result = r.result
        print >> sys.stderr, 'done: %s' % str(params)
        
        n_remaining -= 1
        print 'n_remaining=%d' % n_remaining
    elif r.is_failed:
        print >> sys.stderr, 'failed'
    else:
        results.append(r)

# --
# (Maybe) kill all the workers




