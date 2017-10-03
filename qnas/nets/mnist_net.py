#!/usr/bin/env python

"""
    mnist_net.py
    
    simple network for MNIST classification
"""

import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

class MNISTNet(nn.Module):
    def __init__(self, input_shape=(28, 28)):
        super(MNISTNet, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # Fully connected layers
        tmp = Variable(torch.zeros((1, 1, input_shape[0], input_shape[1])))
        self.fc1_size = np.prod(self.conv(tmp).size()[1:])
        
        self.fc1 = nn.Linear(self.fc1_size, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self.fc1_size)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
    
    def train_step(self, data, targets, optimizer):
        optimizer.zero_grad()
        outputs = self(data)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()
        return outputs, loss.data[0]
