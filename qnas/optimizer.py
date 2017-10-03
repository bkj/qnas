#!/usr/bin/env python

"""
    optimizer.py
    
    *_space arrays give operands/operations as defined in the paper
"""

from functools import partial

import torch
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer

# --
# Operands

operand_space = [
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
]

class Operands(object):
    
    @staticmethod
    def grad_power(grad, state, params):
        power = params
        return grad ** power, state
    
    @staticmethod
    def grad_expavg(grad, state, params):
        """
            !! If two legs of the optimizer use this, it'll get messed up
        """
        power, beta = params
        k = 'grad_expavg_%d' % power
        
        if k not in state:
            state[k] = grad.clone()
        else:
            state[k].mul_(beta).add_(1 - beta, grad ** power)
        
        return state[k], state
    
    @staticmethod
    def grad_sign(grad, state, params):
        power = params
        return torch.sign(grad ** power), state
    
    @staticmethod
    def grad_expavg_sign(grad, state, params):
        x, state = Operands.grad_expavg(grad, params, state)
        return torch.sign(x), state
    
    @staticmethod
    def const(grad, state, params):
        const = params
        return const, state
    
    @staticmethod
    def gaussian_noise(grad, state, params):
        mean, std = params
        return torch.randn(grad.size(), mean, std), state
    
    @staticmethod
    def compound(grad, state, params):
        op1 = partial(getattr(Operands, params['op1'][0]), params=params['op1'][1])
        un1 = partial(getattr(UnaryOperation, params['un1'][0]), params=params['un1'][1])
        
        op2 = partial(getattr(Operands, params['op2'][0]), params=params['op2'][1])
        un2 = partial(getattr(UnaryOperation, params['un2'][0]), params=params['un2'][1])
        
        bin_ = partial(getattr(BinaryOperation, params['bin'][0]), params=params['bin'][1])
        
        left, state = op1(grad, state)
        right, state = op2(grad, state)
        return bin_(un1(left), un2(right)), state

# --
# Unary operations

unary_operation_space = [
    ('identity', None),
    ('negation', None),
    ('exp', None),
    ('abs_log', None),
    ('abs_pow', 0.5),
    ('clip', 1e-5),
    ('clip', 1e-4),
    ('clip', 1e-3),
    ('drop', 0.1),
    ('drop', 0.3),
    ('drop', 0.5),
    ('sign', None),
]


class UnaryOperation(object):
    
    @staticmethod
    def identity(x, params):
        return x
    
    @staticmethod
    def negation(x, params):
        return -x
    
    @staticmethod
    def exp(x, params):
        return torch.exp(x)
    
    @staticmethod
    def abs_log(x, params):
        return torch.exp(torch.abs(x))
    
    @staticmethod
    def abs_pow(x, params):
        power = params
        return torch.pow(torch.abs(x), power)
    
    @staticmethod
    def clip(x, params):
        bound = params
        return torch.clamp(x, -bound, bound)
    
    @staticmethod
    def drop(x, params):
        prob = params
        return F.dropout(x, prob)
    
    @staticmethod
    def sign(x, params):
        return torch.sign(x)

# --
# Binary operations

binary_operation_space = [
    ('add', None),
    ('sub', None),
    ('mul', None),
    ('div', 1e-8),
    ('exp', None),
    ('keep', None),
]

class BinaryOperation(object):
    @staticmethod
    def add(x, y, params):
        return x + y
    
    @staticmethod
    def sub(x, y, params):
        return x - y
    
    @staticmethod
    def mul(x, y, params):
        return x * y
    
    @staticmethod
    def div(x, y, params):
        delta = params
        return x / (y + delta)
    
    @staticmethod
    def exp(x, y, params):
        return x ** y
        
    @staticmethod
    def keep_left(x, y, params):
        return x


# --
# Configurable optimizer
# !! not done yet, at all

class ConfigurableOptimizer(Optimizer):
    def __init__(self, params, config, lr=0.1):
        
        self.op1 = partial(getattr(Operands, config['op1'][0]), params=config['op1'][1])
        self.un1 = partial(getattr(UnaryOperation, config['un1'][0]), params=config['un1'][1])
        
        self.op2 = partial(getattr(Operands, config['op2'][0]), params=config['op2'][1])
        self.un2 = partial(getattr(UnaryOperation, config['un2'][0]), params=config['un2'][1])
        
        self.bin = partial(getattr(BinaryOperation, config['bin'][0]), params=config['bin'][1])
        
        super(ConfigurableOptimizer, self).__init__(params, {
            "lr" : lr,
        })
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]
                
                left, state = self.op1(grad, state)
                right, state = self.op2(grad, state)
                update = self.bin(self.un1(left), self.un2(right))
                
                p.data.add_(-group['lr'], update)
        
        return loss

# --
# Testing

import sys
from torch.autograd import Variable
from data import DataLoader
from nets.two_layer_net import TwoLayerNet
from helpers import to_numpy

sys.path.append('../docker')
from worker import GridPointWorker

configs = {
    'sgd' : {
        "op1" : ('grad_power', 1),
        "un1" : ('identity', None),
        
        "op2" : ('const', 1),
        "un2" : ('identity', None),
        
        "bin" : ('mul', None)
    },
    'powersign' : {
        'op1' : ('compound', {
            "op1" : ('grad_power', 1),
            "un1" : ('sign', None),
            
            "op2" : ('grad_expavg', (1, 0.9)),
            "un2" : ('sign', None),
            
            "bin" : ('mul', None),
        }),
        'un1' : ('exp', None),
        
        'op2' : ('grad_power', 1),
        'un2' : ('identity', None),
        
        'bin' : ('mul', None),
    }
}

ds = DataLoader(root='../data/', pin_memory=False, num_workers=2).CIFAR10()
loader = ds['train_loader']

net = TwoLayerNet().cuda()
opt = ConfigurableOptimizer(net.parameters(), configs['sgd'])
lr_scheduler = lambda x: 0.01

epochs = 5
for epoch in range(epochs):
    print 'epoch=%d' % epoch
    curr_acc, curr_loss, history = GridPointWorker._train_epoch(net, loader, opt, epoch, lr_scheduler, ds['n_train_batches'])
    val_acc, val_loss = GridPointWorker._eval(net, epoch, ds['val_loader'], mode='val')
    print val_acc

# powersign -> 0.6428 @ 5epoch
# sgd -> 
