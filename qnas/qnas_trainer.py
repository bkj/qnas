#!/usr/bin/env python

"""
    worker.py
    
    !! This could certainly be simplified -- too much redundant boilerplate
"""

from __future__ import division

import os
import sys
import time
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
cudnn.benchmark = True

from lr import LRSchedule
from data import DataLoader
from nets import net_constructors

# --
# Training helpers

class QNASTrainer(object):
    def __init__(self, config, lr_fail_factor=0.5, max_failures=3, cuda=True, dataset_num_workers=2):
    
        # Model parameters
        self.config    = config
        self.net_class = config['net_class']
        self.dataset   = config['dataset']
        self.epochs    = config['epochs']
        
        # Execution parameters
        self.lr_fail_factor      = lr_fail_factor
        self.max_failures        = max_failures
        self.cuda                = cuda
        self.dataset_num_workers = dataset_num_workers
        
        self.fail_counter = 0
        self.hist = []
        
        self._setup()
        
    def _setup(self):
        # Reset epoch count
        self.epoch = 0
        
        # Set dataset
        self.ds = getattr(DataLoader(root='../data/', pin_memory=self.cuda, 
            num_workers=self.dataset_num_workers), self.dataset)()
        
        # Network
        self.net = net_constructors[self.net_class](self.config, num_classes=self.ds['num_classes'])
        if self.cuda:
            self.net = self.net.cuda()
    
    @staticmethod
    def _train_epoch(net, loader, epoch, n_train_batches):
        _ = net.train()
        all_loss, correct, total = 0, 0, 0
        history = []
        gen = tqdm(enumerate(loader), total=n_train_batches)
        for batch_idx, (data, targets) in gen:
            if next(net.parameters()).is_cuda:
                data, targets = data.cuda(), targets.cuda()
            
            data, targets = Variable(data), Variable(targets)
            
            outputs, loss = net.train_step(data, targets, progress=epoch + batch_idx / n_train_batches)
            
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
            
            if batch_idx > n_train_batches:
                break
        
        return curr_acc, curr_loss, history
    
    @staticmethod
    def _eval(net, loader, epoch, mode='val'):
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
    
    def _train(self):
        while self.epoch < self.epochs:
            
            # Train for an epoch
            train_acc, train_loss, train_history = QNASTrainer._train_epoch(
                net=self.net,
                loader=self.ds['train_loader'],
                epoch=self.epoch,
                n_train_batches=self.ds['n_train_batches'],
            )
            
            # Sometimes the models don't converge...
            if np.isnan(train_loss):
                self.fail_counter += 1
                if self.fail_counter <= self.max_failures:
                    print >> sys.stderr, 'grid-point.py: train_loss is NaN -- reducing LR and restarting'
                    self.config['lr_init'] *= self.lr_fail_factor
                    self._setup()
                    continue
                else:
                    print >> sys.stderr, 'grid-point.py: train_loss is NaN -- too many failures -- exiting'
                    return self.hist
            
            # Eval on val + test datasets
            val_acc, val_loss = QNASTrainer._eval(self.net, self.ds['val_loader'], self.epoch,  mode='val')
            test_acc, test_loss = QNASTrainer._eval(self.net, self.ds['test_loader'], self.epoch, mode='test')
            
            # Log performance
            self.hist.append({
                'epoch'              : self.epoch, 
                'lr'                 : self.net.lr,
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

def qnas_trainer_run(config, **kwargs):
    qtrainer = QNASTrainer(config, **kwargs)
    results = qtrainer._train()
    qtrainer.save()
    return qtrainer.config, results