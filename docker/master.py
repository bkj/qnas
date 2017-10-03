#!/usr/bin/env python

"""
    qnas-master.py
"""

import sys
import time
from tqdm import tqdm
from rq import Queue
from redis import Redis
from collections import deque

from worker import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=str, default='qnas-0')
    parser.add_argument('--redis-password', type=str)
    parser.add_argument('--redis-host', type=str)
    parser.add_argument('--redis-port', type=int)
    parser.add_argument('--result-ttl', type=int, default=60 * 60 * 2) # 2 hours
    parser.add_argument('--ttl', type=int, default=60 * 60 * 2) # 2 hours
    return parser.parse_args()


def callback(config, result, args, failed=False):
    """ action to take when results are returned """
    if not failed:
        print >> sys.stderr, 'job finished: %s' % json.dumps(config)
        
        result_path = os.path.join('results', args.run, config['model_name'])
        if not os.path.exists(os.path.dirname(result_path)):
            os.makedirs(os.path.dirname(result_path))
        
        open(result_path, 'w').write('\n'.join(map(json.dumps, result)))
    else:
        print >> sys.stderr, 'job failed: %s' % json.dumps(config)


if __name__ == "__main__":
    
    args = parse_args()
    
    c = Redis(host=args.redis_host, port=args.redis_port, password=args.redis_password)
    
    results = deque()
    q = Queue(connection=Redis(host=args.redis_host, port=args.redis_port, password=args.redis_password))
    q.empty() # Clear queue -- could be dangerous
    
    config = {
        "op_keys":["double_bnconv_3","identity","add"],
        "red_op_keys":["conv_1","double_bnconv_3","add"],
        "model_name":"test"
    }
    
    # Make sure model names are unique
    n_jobs = 24
    for _ in tqdm(range(n_jobs)):
        time.sleep(0.01)
        results.append(q.enqueue(run_job, config, epochs=1, cuda=False, ttl=args.ttl, result_ttl=args.result_ttl))
    
    # Run jobs, executing callback at each one
    while True:
        if len(results) == 0:
            break
        
        r = results.popleft()
        
        if r.is_finished:
            callback(params, result, args, failed=False)
        elif r.is_failed:
            callback(params, result, args, failed=True)
        else:
            results.append(r)
