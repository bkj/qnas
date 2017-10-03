#!/usr/bin/env python

"""
    qnas-master.py
"""

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
    
    parser.add_argument('--result-ttl', type=int, default=60 * 60 * 6) # 6 hours
    parser.add_argument('--ttl', type=int, default=60 * 60 * 6) # 6 hours
    parser.add_argument('--timeout', type=int, default=60 * 60 * 6) # 6 hours
    
    return parser.parse_args()


def callback(config, result, args):
    """ action to take when results are returned """
    print >> sys.stderr, 'job finished: %s' % json.dumps(config)
    
    result_path = os.path.join('results', args.run, config['model_name'])
    if not os.path.exists(os.path.dirname(result_path)):
        os.makedirs(os.path.dirname(result_path))
    
    open(result_path, 'w').write('\n'.join(map(json.dumps, result)))


if __name__ == "__main__":
    
    args = parse_args()
    
    c = Redis(host=args.redis_host, port=args.redis_port, password=args.redis_password)
    
    results = deque()
    q = Queue(connection=Redis(host=args.redis_host, port=args.redis_port, password=args.redis_password))
    q.empty() # Clear queue -- could be dangerous
    
    # Make sure model names are unique
    n_jobs = 48
    for i in tqdm(range(n_jobs)):
        time.sleep(0.01)
        
        # config = {
        #     "op_keys":["double_bnconv_3","identity","add"],
        #     "red_op_keys":["conv_1","double_bnconv_3","add"],
        #     "model_name":"test",
        # }
        config = {"model_name" : "test-%d" % i}
        r = q.enqueue(
            run_job,
            config=config, net_class='mnist_net', dataset='MNIST', cuda=False, epochs=1,
            ttl=args.ttl, result_ttl=args.result_ttl, timeout=args.timeout,
        )
        results.append(r)
    
    # Run jobs, executing callback at each one
    while True:
        if len(results) == 0:
            break
        
        r = results.popleft()
        
        if r.is_finished:
            config, result = r.result
            callback(config, result, args)
        elif r.is_failed:
            print >> sys.stderr, 'failed!'
        else:
            results.append(r)
    
    # Kill the workers -- a little hacky, there's probably a better way
    n_kills = 250
    for _ in range(n_kills):
        _ = q.enqueue(kill, ttl=args.ttl, timeout=args.timeout)
        
