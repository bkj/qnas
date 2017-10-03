#!/usr/bin/env python

"""
    two_layer_net.py
    
    Implemented to the specs of "a small two layer 3x3 ConvNet" per the Bello et. al paper:
        | The child network architecture that all sampled optimizers
        | are run on is a small two layer 3x3 ConvNet. This ConvNet
        | has 32 filters with ReLU activations and batch normalization
        | applied after each convolutional layer.    
"""

import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

class TwoLayerNet(nn.Module):
    def __init__(self, num_classes=10, input_channels=3, hidden_channels=32, input_shape=(32, 32)):
        super(TwoLayerNet, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_channels),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_channels),
            nn.Linear(hidden_channels, num_classes)
        )
    
    def forward(self, x):
        return self.conv(x)
    
    def train_step(self, data, targets, optimizer):
        optimizer.zero_grad()
        outputs = self(data)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()
        return outputs, loss.data[0]
