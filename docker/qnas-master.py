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

jobs = range(2)

for job in jobs:
    time.sleep(0.01)
    print >> sys.stderr, 'enqueuing %s' % str(job)
    results.append(q.enqueue(run_job, job, ttl=ttl, result_ttl=result_ttl))

# --
# Wait for jobs to finish

while True:
    if len(results) == 0:
        break
    
    r = results.popleft()
    
    if r.is_finished:
        params, result = r.result
        results.append(q.enqueue(run_job, result))
        print >> sys.stderr, 'done: %s' % str(params)
        print 'n_remaining=%d' % len(results)
        
    elif r.is_failed:
        print >> sys.stderr, 'failed'
    else:
        results.append(r)


