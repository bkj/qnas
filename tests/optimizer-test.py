#!/usr/bin/env python

"""
    optimizer-test.py
"""

import sys
sys.path.append('../qnas')
from data import DataLoader
from helpers import to_numpy
from trainer import QNASTrainer
from nets.optnet import OptNetSmall
from optimizer import ConfigurableOptimizer

from controllers.optimizer_search import OptimizerSampler

# --
# Define optimizers

archs = {
    'sgd' : {
        "op1" : ('grad_power', 1),
        "un1" : 'identity',
        
        "op2" : ('const', 1),
        "un2" : 'identity',
        
        "bin" : 'mul',
    },
    'powersign' : {
        'op1' : ('compound', {
            "op1" : ('grad_power', 1),
            "un1" : 'sign',
            
            "op2" : ('grad_expavg', (1, 0.9, '*')),
            "un2" : 'sign',
            
            "bin" : 'mul',
        }),
        'un1' : 'exp',
        
        'op2' : ('grad_power', 1),
        'un2' : 'identity',
        
        'bin' : 'mul',
    },
    'addsign' : {
        'op1' : ('compound', {
            "op1" : ('compound', {
                "op1" : ('grad_power', 1),
                "un1" : 'sign',
                
                "op2" : ('grad_expavg', (1, 0.9, '*')),
                "un2" : 'sign',
                
                "bin" : 'mul',
            }),
            "un1" : 'identity',
            
            "op2" : ('const', 1),
            "un2" : 'identity',
            
            "bin" : 'add',
        }),
        'un1' : 'identity',
        
        'op2' : ('grad_power', 1),
        'un2' : 'identity',
        
        'bin' : 'mul',
    },
}

# --
# "Unit tests"

import torch
from optimizer import Operands, UnaryOperation, BinaryOperation

# Operands
grad = torch.rand((10, 10))
state = {}
for _ in range(3):
    _ = Operands.grad_power(grad, state, 1)
    _ = Operands.grad_power(grad, state, 2)
    _ = Operands.grad_power(grad, state, 3)
    _ = Operands.grad_expavg(grad, state, (1, 0.9, 'placeholder'))
    _ = Operands.grad_expavg(grad, state, (2, 0.999, 'placeholder'))
    _ = Operands.grad_expavg(grad, state, (3, 0.999, 'placeholder'))
    _ = Operands.const(grad, state, 1)
    _ = Operands.gaussian_noise(grad, state, (0, 1))

# Unary operations
grad = torch.randn((10, 10))
_ = UnaryOperation.identity(grad, None)
_ = UnaryOperation.negation(grad, None)
_ = UnaryOperation.exp(grad, None)
_ = UnaryOperation.abs_log(grad, None)
_ = UnaryOperation.abs_pow(grad, 0.5)
_ = UnaryOperation.clip(grad, 0.5)
_ = UnaryOperation.drop(grad, 0.5)
_ = UnaryOperation.sign(grad, None)

# Binary operations
grad_a = torch.randn((10, 10))
grad_b = torch.randn((10, 10))
_ = BinaryOperation.add(grad_a, grad_b, None)
_ = BinaryOperation.sub(grad_a, grad_b, None)
_ = BinaryOperation.mul(grad_a, grad_b, None)
_ = BinaryOperation.div(grad_a, grad_b, 1e-8)
_ = BinaryOperation.keep_left(grad_a, grad_b, None)

# --
# Run

arch = archs['powersign']

ds = DataLoader(root='../data/', pin_memory=False, num_workers=2).CIFAR10()

net = OptNetSmall({
    "lr_schedule" : 'constant',
    "epochs" : 1,
    "lr_init" : 0.01,
    "opt_arch" : arch
}).cuda()

epoch = 0
print 'epoch=%d' % epoch
curr_acc, curr_loss, history = QNASTrainer._train_epoch(net, ds['train_loader'], epoch, ds['n_train_batches'])
val_acc, val_loss = QNASTrainer._eval(net, ds['val_loader'], epoch, mode='val')
print val_acc
print


arch = {'bin': 'keep', 'un1': 'exp', 'un2': ('drop', 0.1), 'op1': ('grad_expavg', (3, 0.999)), 'op2': ('grad_sign', 1)}

ds = DataLoader(root='../data/', pin_memory=False, num_workers=2).CIFAR10()

net = TwoLayerNet({
    "lr_schedule" : 'constant',
    "epochs" : 1,
    "lr_init" : 0.01
}).cuda()



curr_acc, curr_loss, history = QNASTrainer._train_epoch(net, ds['train_loader'], epoch, ds['n_train_batches'])

# --
# Results
# 
# powersign -> 0.6428 @ 5epoch
# sgd       -> 0.6044 @ 5epoch
# addsign   -> 0.6508 @ 5epoch