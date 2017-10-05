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

from ..lr import LRSchedule

class TwoLayerNet(nn.Module):
    def __init__(self, config, num_classes=10, input_channels=3, hidden_channels=32, input_shape=32):
        super(TwoLayerNet, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_channels),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_channels),
        )
        
        self.fc = nn.Linear(hidden_channels * (input_shape - 4) ** 2, num_classes)
        
        # Optimizer
        self.lr_scheduler = functools.partial(getattr(LRSchedule, config['lr_schedule']), 
            lr_init=config['lr_init'], epochs=config['epochs'])
        
        self.lr = self.lr_scheduler(0.0)
        
        self.opt = torch.optim.SGD(self.parameters(), lr=self.lr, 
            momentum=0.9, weight_decay=5e-4)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
        
    def train_step(self, data, targets, progress):
        self.lr = self.lr_scheduler(progress)
        LRSchedule.set_lr(self.opt, self.lr)
        
        self.opt.zero_grad()
        outputs = self(data)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        self.opt.step()
        return outputs, loss.data[0]


if __name__ == "__main__":
    net = TwoLayerNet()
    data = Variable(torch.zeros((128, 3, 32, 32)))
    z = net(data)
    print(z.size(), file=sys.stderr)