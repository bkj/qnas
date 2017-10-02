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

# --
# Params 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=str, default='qnas-0')
    parser.add_argument('--result-ttl', type=int, default=60 * 60 * 2) # 2 hours
    parser.add_argument('--ttl', type=int, default=60 * 60 * 2) # 2 hours
    return parser.parse_args()

# --
# Helpers



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


def callback(config, result):
    result_path = os.path.join('results', run, config['model_name'])
    if not os.path.exists(os.path.dirname(result_path)):
        os.makedirs(os.path.dirname(result_path))
    
    open(result_path, 'w').write('\n'.join(map(json.dumps, result)))


if __name__ == "__main__":
    
    results = deque()
    q = Queue(connection=Redis())
    q.empty() # Clear queue -- could be dangerous
    
    config = {"op_keys":["double_bnconv_3","identity","add"],"red_op_keys":["conv_1","double_bnconv_3","add"],"model_name":"test"}
    
    for _ in range(2):
        time.sleep(0.01)
        print >> sys.stderr, 'enqueuing %s' % str(job)
        results.append(q.enqueue(run_job, config, epochs=1, ttl=ttl, result_ttl=result_ttl))
    
    # Enqeue jobs
    results = initialize()
    
    # Run jobs, executing callback at each one
    run(results, callback)