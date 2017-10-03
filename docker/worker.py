#!/usr/bin/env python

"""
    worker.py
    
    !! This might be more complicated than it needs to be...
"""

from __future__ import division

import os
import sys
import json
import time
import base64
import argparse
import functools
import numpy as np
from tqdm import tqdm
from hashlib import md5
from datetime import datetime
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
cudnn.benchmark = True

import torchvision
import torchvision.transforms as transforms

sys.path.append('..')
from nas import RNet
from data import DataLoader
from lr import LRSchedule

# --
# Network constructor helpers

class MNISTNet(nn.Module):
    def __init__(self, input_shape=(28, 28)):
        super(MNISTNet, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # nn.Dropout2d(p=0.25),
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


class NetConstructors(object):
    @staticmethod
    def rnet(config, num_classes, cuda=True):
        net = RNet(
            config['op_keys'],
            config['red_op_keys'],
            num_classes=num_classes,
        )
        
        if cuda:
            net = net.cuda()
        
        return net
    
    @staticmethod
    def mnist_net(config, num_classes, cuda=True):
        net = MNISTNet()
        
        if cuda:
            net = net.cuda()
        
        return net

# --
# Training helpers

class GridPointWorker(object):
    
    def __init__(self, config, net_class='rnet', dataset='CIFAR10', epochs=20, 
        lr_schedule='linear', lr_init=0.1, lr_fail_factor=0.5, max_failures=3, cuda=True):
    
        self.config = config
        self.config['vars'] = {
            "dataset"        : dataset,
            "epochs"         : epochs,
            "lr_schedule"    : lr_schedule,
            "lr_init"        : lr_init,
            "lr_fail_factor" : lr_fail_factor,
            "max_failures"   : max_failures,
        }
        
        self.dataset        = dataset
        self.epoch          = 0
        self.epochs         = epochs
        self.lr_schedule    = lr_schedule
        self.lr_init        = lr_init
        self.fail_counter   = 0
        self.lr_fail_factor = lr_fail_factor
        self.max_failures   = max_failures
        
        self.hist = []
        
        # Set dataset
        self.ds = getattr(DataLoader(root='../data/', pin_memory=cuda, num_workers=2), dataset)()
        
        # Define network
        self.net = getattr(NetConstructors, net_class)(config, self.ds['num_classes'], cuda=cuda)
        
        # Set LR scheduler
        self.lr_scheduler = functools.partial(getattr(LRSchedule, lr_schedule), lr_init=lr_init, epochs=epochs)
        
        # Set optimizer
        self.opt = optim.SGD(self.net.parameters(), lr=self.lr_scheduler(float(self.epoch)), momentum=0.9, weight_decay=5e-4)
    
    @staticmethod
    def _train_epoch(net, loader, opt, epoch, lr_scheduler, n_train_batches):
        _ = net.train()
        all_loss, correct, total = 0, 0, 0
        history = []
        gen = tqdm(enumerate(loader), total=len(loader))
        for batch_idx, (data, targets) in gen:
            LRSchedule.set_lr(opt, lr_scheduler(epoch + batch_idx / n_train_batches))
            
            if next(net.parameters()).is_cuda:
                data, targets = data.cuda(), targets.cuda()
            
            data, targets = Variable(data), Variable(targets)
            outputs, loss = net.train_step(data, targets, opt)
            
            if np.isnan(loss):
                return np.nan, np.nan, []
            
            predicted = outputs.data.max(1)[1]
            batch_correct = (predicted == targets.data).cpu().sum()
            batch_size = targets.size(0)
            
            history.append((loss, batch_correct / batch_size))
            
            total += batch_size
            correct += batch_correct
            all_loss += loss
            
            curr_acc = correct / total
            curr_loss = all_loss / (batch_idx + 1)
            gen.set_postfix(OrderedDict([('epoch', epoch), ('train_loss', curr_loss), ('train_acc', curr_acc)]))
        
        return curr_acc, curr_loss, history

    @staticmethod
    def _eval(net, epoch, loader, mode='val'):
        _ = net.eval()
        all_loss, correct, total = 0, 0, 0
        gen = tqdm(enumerate(loader), total=len(loader))
        for batch_idx, (data, targets) in gen:
            
            if next(net.parameters()).is_cuda:
                data, targets = data.cuda(), targets.cuda()
            
            data, targets = Variable(data, volatile=True), Variable(targets)
            outputs = net(data)
            loss = F.cross_entropy(outputs, targets).data[0]
            
            predicted = outputs.data.max(1)[1]
            batch_correct = (predicted == targets.data).cpu().sum()
            batch_size = targets.size(0)
            
            total += batch_size
            correct += batch_correct
            all_loss += loss
                    
            curr_acc = correct / total
            curr_loss = all_loss / (batch_idx + 1)
            gen.set_postfix(OrderedDict([('epoch', epoch), ('%s_loss' % mode, curr_loss), ('%s_acc' % mode, curr_acc)]))
        
        return curr_acc, curr_loss
    
    
    def run(self):
        while self.epoch < self.epochs:
            
            # Train
            train_acc, train_loss, train_history = GridPointWorker._train_epoch(
                net=self.net,
                loader=self.ds['train_loader'],
                opt=self.opt,
                epoch=self.epoch,
                lr_scheduler=self.lr_scheduler,
                n_train_batches=self.ds['n_train_batches'],
            )
            
            # Sometimes the models don't converge...
            # This is just a heuristic, may not lead to the fairest comparisons
            if np.isnan(train_loss):
                self.fail_counter += 1
                if self.fail_counter <= self.max_failures:
                    print >> sys.stderr, 'grid-point.py: train_loss is NaN -- reducing LR and restarting'
                    
                    # Reset epoch count
                    self.epoch = 0
                    
                    # Reduce initial learning rate
                    self.lr_init *= self.lr_fail_factor
                    
                    # Set LR scheduler
                    self.lr_scheduler = functools.partial(getattr(LRSchedule, self.lr_schedule), lr_init=self.lr_init, epochs=self.epochs)
                    
                    # (Re)define network
                    self.net = getattr(NetConstructors, net_class)(config, self.ds['num_classes'], cuda=cuda)
                    
                    # (Re)define optimizer
                    self.opt = optim.SGD(self.net.parameters(), lr=self.lr_scheduler(float(self.epoch)), momentum=0.9, weight_decay=5e-4)
                    continue
                else:
                    print >> sys.stderr, 'grid-point.py: train_loss is NaN -- too many failures -- exiting'
                    os._exit(0)
            
            # Eval
            val_acc, val_loss = GridPointWorker._eval(self.net, self.epoch, self.ds['val_loader'], mode='val')
            test_acc, test_loss = GridPointWorker._eval(self.net, self.epoch, self.ds['test_loader'], mode='test')
            
            # Log
            self.hist.append({
                'epoch'              : self.epoch, 
                'lr'                 : self.lr_scheduler(self.epoch + 1),
                'timestamp'          : datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
                
                'train_acc'          : train_acc, 
                'train_loss'         : train_loss,
                'train_history'      : train_history,
                
                'val_acc'            : val_acc, 
                'val_loss'           : val_loss,
                
                'test_acc'           : test_acc,
                'test_loss'          : test_loss,
            })
            
            self.epoch += 1
            print
        
        return self.hist
    
    def save(self):
        torch.save(self.net.state_dict(), self.config['model_name'])

# --
# RQ helpers

def run_job(config, **kwargs):
    gpworker = GridPointWorker(config, **kwargs)
    results = gpworker.run()
    gpworker.save()
    return gpworker.config, results


def run_dummy(config, **kwargs):
    time.sleep(3)
    return config, {"dummy" : True}


def kill(delay=1):
    print >> sys.stderr, '!!! Kill Signal Received -- shutting down in %ds' % delay
    time.sleep(delay)
    os.kill(os.getppid(), 9)

