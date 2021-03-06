#!/usr/bin/env python

"""
    controllers/optimizer_search.py
"""

import sys
import json
import random
from uuid import uuid4

optimizer_space = {
    "op" : [
        ('grad_power', 1),
        ('grad_power', 2),
        ('grad_power', 3),
        
        ('grad_expavg', (1, 0.9, 'placeholder')), # placeholder for stateful
        ('grad_expavg', (2, 0.999, 'placeholder')), # placeholder for stateful
        ('grad_expavg', (3, 0.999, 'placeholder')), # placeholder for stateful
        
        ('grad_sign', 1),
        ('grad_expavg_sign', (1, 0.9, 'placeholder')), # placeholder for stateful
        
        ('const', 1),
        ('const', 2),
        
        # ('gaussian_noise', (0, 0.01)), # !! Slow...
        
        # ('weight_power', -1), # !! What are these?  The weights themselves raised to some power?
        # ('weight_power', -2),
        # ('weight_power', -3),
        # ('weight_power', -4),
        
        # ('adam', adam_config),       # !! What does using another optimizer mean exactly?
        # ('rmsprop', rmsprop_config), # ^^
    ],
    "un" : [
        'identity',
        'negation',
        'exp',
        'abs_log',
        ('abs_pow', 0.5),
        ('clip', 1e-5),
        ('clip', 1e-4),
        ('clip', 1e-3),
        ('drop', 0.1),
        ('drop', 0.3),
        ('drop', 0.5),
        'sign',
    ],
    "bin" : [
        'add',
        'avg',
        'sub',
        'mul',
        ('div', 1e-8),
        # 'exp',
        'keep_left',
    ]
}

class OptimizerSampler(object):
    def sample(self, depth=0, seed=None):
        if seed:
            random.seed(123)
        
        return {
            "op1" : random.choice(optimizer_space['op']) if not depth else ('compound', self.sample(depth - 1)),
            "un1" : random.choice(optimizer_space['un']),
            
            "op2" : random.choice(optimizer_space['op']),
            "un2" : random.choice(optimizer_space['un']),
            
            "bin" : random.choice(optimizer_space['bin'])
        }


class RandomOptimizerController(object):
    def __init__(self, depth=1):
        self.depth = depth
        self.sampler = OptimizerSampler()
    
    def _next(self, last=None):
        return {
            "func" : "qnas_trainer",
            "config" : {
                "model_name"  : "opt-test2-%s" % str(uuid4()),
                "net_class"   : 'OptNetSmall',
                "dataset"     : 'CIFAR10',
                "epochs"      : 1,
                "lr_schedule" : 'constant',
                "lr_init"     : 0.01,
                "opt_arch"    : self.sampler.sample(depth=self.depth),
            },
            "cuda" : True,
            "lr_fail_factor" : 0.5,
            "dataset_num_workers" : 8
        }
    
    def seed(self):
        return self._next()
    
    def success(self, last):
        config, hist = last
        print >> sys.stderr, 'job finished: %s' % json.dumps(config)
        open('./results/hists/%s' % config['model_name'], 'w').write('\n'.join(map(json.dumps, hist)))
        open('./results/configs/%s' % config['model_name'], 'w').write(json.dumps(config))
        return self._next()
    
    def failure(self, last):
        print >> sys.stderr, 'job failed!'
        return self._next()

