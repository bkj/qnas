#!/usr/bin/env python

"""
    controllers/optimizer_search.py
"""

from __future__ import print_function

import os
import sys
import json
import random
from glob import glob
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

# --

class BaseOptimizerController(object):
    def __init__(self, **kwargs):
        
        self.configs_dir = os.path.join(kwargs['outdir'], kwargs['run_name'], 'configs')
        if not os.path.exists(self.configs_dir):
            os.makedirs(self.configs_dir)
            
        self.hists_dir = os.path.join(kwargs['outdir'], kwargs['run_name'], 'hists')
        if not os.path.exists(self.hists_dir):
            os.makedirs(self.hists_dir)
        
        self.run_name = kwargs['run_name']


# --

class RandomOptimizerSampler(object):
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


class RandomOptimizerController(BaseOptimizerController):
    def __init__(self, depth=0, **kwargs):
        
        super(RandomOptimizerController, self).__init__(**kwargs)
        
        self.depth = depth
        self.random_sampler = RandomOptimizerSampler()
    
    def _next(self, last=None):
        return {
            "func" : "qnas_trainer",
            "config" : {
                "model_name"  : "%s-%s" % (self.run_name, str(uuid4())),
                "net_class"   : 'OptNetSmall',
                "dataset"     : 'CIFAR10',
                "epochs"      : 1,
                "lr_schedule" : 'constant',
                "lr_init"     : 0.01,
                "opt_arch"    : self.random_sampler.sample(depth=self.depth),
            },
            "cuda" : True,
            "lr_fail_factor" : 0.5,
            "dataset_num_workers" : 8
        }
    
    def seed(self):
        return self._next()
    
    def success(self, last):
        config, hist = last
        print('job finished: %s' % json.dumps(config), file=sys.stderr)
        
        # Save results
        open(os.path.join(self.configs_dir, config['model_name']), 'w').write(json.dumps(config))
        open(os.path.join(self.hists_dir, config['model_name']), 'w').write('\n'.join(map(json.dumps, hist)))
        
        return self._next()
    
    def failure(self, last):
        print('job failed!', file=sys.stderr)
        return self._next()

# --

class EnumeratedOptimizerController(BaseOptimizerController):
    def __init__(self, indir=None, **kwargs):
        assert(indir is not None)
        
        super(EnumeratedOptimizerController, self).__init__(**kwargs)
        
        self.configs = [json.load(open(f)) for f in glob(indir)]
    
    def is_empty(self):
        return len(self.configs) == 0
    
    def seed(self):
        return {
            "func" : "qnas_trainer",
            "config" : {
                "model_name"  : "%s-%s" % (self.run_name, str(uuid4())),
                "net_class"   : 'OptNetSmall',
                "dataset"     : 'CIFAR10',
                "epochs"      : 1,
                "lr_schedule" : 'constant',
                "lr_init"     : 0.01,
                "opt_arch"    : self.configs.pop(0),
            },
            "cuda" : True,
            "lr_fail_factor" : 0.5,
            "dataset_num_workers" : 8
        }
    
    def success(self, last):
        config, hist = last
        print('job finished: %s' % json.dumps(config), file=sys.stderr)
        
        # Save results
        open(os.path.join(self.configs_dir, config['model_name']), 'w').write(json.dumps(config))
        open(os.path.join(self.hists_dir, config['model_name']), 'w').write('\n'.join(map(json.dumps, hist)))
    
    def failure(self, last):
        print('job failed!', file=sys.stderr)

# --

class PPOOptimizerSampler(object):
    def __init__(self, depth=0):
        self.depth = 0
    
    def update(self, arch, reward):
        pass
        # --
        # Update model
        # self.net.update(arch, reward)
        
    def sample(self):
        pass
        # --
        # Sample from policy
        # return self.net.sample()


class PPOOptimizerController(object):
    def __init__(self, depth=1, **kwargs):
        self.configs_dir = os.path.join(kwargs['outdir'], kwargs['run_name'], 'configs')
        if not os.path.exists(self.configs_dir):
            os.makedirs(self.configs_dir)
            
        self.hists_dir = os.path.join(kwargs['outdir'], kwargs['run_name'], 'hists')
        if not os.path.exists(self.hists_dir):
            os.makedirs(self.hists_dir)
        
        self.run_name = kwargs['run_name']
        
        self.depth = depth
        self.ppo_sampler = PPOOptimizerSampler()
        self.random_sampler = RandomOptimizerSampler(self.depth)
        
    def _wrap(self, arch):
        return {
            "func" : "qnas_trainer",
            "config" : {
                "model_name"  : "%s-%s" % (self.run_name, str(uuid4())),
                "net_class"   : 'OptNetSmall',
                "dataset"     : 'CIFAR10',
                "epochs"      : 1,
                "lr_schedule" : 'constant',
                "lr_init"     : 0.01,
                "opt_arch"    : arch,
            },
            "cuda" : True,
            "lr_fail_factor" : 0.5,
            "dataset_num_workers" : 8
        }
    
    def seed(self):
        return self._wrap(arch=self.random_sampler(depth=self.depth))
    
    def success(self, last):
        config, hist = last
        print('job finished: %s' % json.dumps(config), file=sys.stderr)
        
        # Save
        open(os.path.join(self.configs_dir, config['model_name']), 'w').write(json.dumps(config))
        
        hist['config'] = config
        open(os.path.join(self.hists_dir, config['model_name']), 'w').write('\n'.join(map(json.dumps, hist)))
        
        return self._next()
    
    def failure(self, last):
        print('job failed!', file=sys.stderr)
        return self._next()

