#!/usr/bin/env python

"""
    controllers/optimizer.py
"""

import sys
import json
import random
from uuid import uuid4
from tqdm import tqdm
from base import BaseController

sys.path.append('..')
from qnas_trainer import qnas_trainer_run

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


class RandomOptimizerController(BaseController, OptimizerSampler):
    def __init__(self, args):
        super(RandomOptimizerController, self).__init__(args)
        
        self.all_hist = []
    
    def initialize(self, depth=1, n_jobs=100):
        for i in tqdm(range(n_jobs)):
            self.enqueue(qnas_trainer_run, {
                "config" : {
                    "model_name"  : "opt-test-%s" % str(uuid4()),
                    "net_class"   : 'OptNetSmall',
                    "dataset"     : 'CIFAR10',
                    "epochs"      : 1,
                    "lr_schedule" : 'constant',
                    "lr_init"     : 0.1,
                    "opt_arch"    : self.sample(depth=depth),
                },
                "cuda" : False,
                "lr_fail_factor" : 0.1,
                "dataset_num_workers" : 8
            })
    
    def callback(self, result):
        config, hist = result
        print >> sys.stderr, 'job finished: %s' % json.dumps(config)
        for h in hist:
            print json.dumps({
                "train_acc" : h['train_acc'], 
                "val_acc" : h['val_acc'],
                "test_acc" : h['test_acc'],
            })
        