#!/usr/bin/env python

"""
    controllers/optimizer.py
"""

optimizer_space = {
    "op" : [
        ('grad_power', 1),
        ('grad_power', 2),
        ('grad_power', 3),
        
        ('grad_expavg', (1, 0.9)),
        ('grad_expavg', (2, 0.999)),
        ('grad_expavg', (3, 0.999)),
        
        ('grad_sign', 1),
        ('grad_expavg_sign', (1, 0.9)),
        
        ('const', 1),
        ('const', 2),
        
        ('gaussian_noise', (0, 0.01)),
        
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
        'sub',
        'mul',
        ('div', 1e-8),
        'exp',
        'keep',
    ]
}

import random
from pprint import pprint

class RandomOptimizerController(object):
    def sample(self, depth=0, seed=None):
        if seed:
            random.seed(123)
        
        return {
            "op1" : random.choice(optimizer_space['op']) if not depth else self.sample(depth - 1),
            "un1" : random.choice(optimizer_space['un']),
            
            "op2" : random.choice(optimizer_space['op']),
            "un2" : random.choice(optimizer_space['un']),
            
            "bin" : random.choice(optimizer_space['bin'])
        }


randopt = RandomOptimizerController()

pprint(randopt.sample(depth=2))

