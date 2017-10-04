#!/usr/bin/env python

"""
    two_layer_net.py
    
    Implemented to the specs of "a small two layer 3x3 ConvNet" per the Bello et. al paper:
        | The child network architecture that all sampled optimizers
        | are run on is a small two layer 3x3 ConvNet. This ConvNet
        | has 32 filters with ReLU activations and batch normalization
        | applied after each convolutional layer.
    
    They don't say what they do for the fully connected layer.  This is the 
    simplest thing, but who knows if there's dropout / whatever
"""

import sys
import functools
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

sys.path.append('..')
from lr import LRSchedule

from two_layer_net import TwoLayerNet
from optimizer import ConfigurableOptimizer

class OptNetSmall(TwoLayerNet):
    def __init__(self, config, **kwargs):
        super(OptNetSmall, self).__init__(config, **kwargs)
        self.opt = ConfigurableOptimizer(self.parameters(), config['opt_arch'])
        
    def train_step(self, data, targets, progress):
        self.lr = self.lr_scheduler(progress)
        LRSchedule.set_lr(self.opt, self.lr)
        
        self.opt.zero_grad()
        outputs = self(data)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        self.opt.step()
        return outputs, loss.data[0]
