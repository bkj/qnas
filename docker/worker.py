#!/usr/bin/env python

"""
    worker.py
"""

from __future__ import division

import os
import sys
import time
import argparse
import functools
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

sys.path.append('../qnas')
sys.path.append('../qnas/nets')
from lr import LRSchedule
from data import DataLoader

from nets.rnet import RNet
from nets.mnist_net import MNISTNet
from nets.two_layer_net import TwoLayerNet

# --
# Network constructor helpers

class NetConstructors(object):
    @staticmethod
    def rnet(config, num_classes):
        return RNet(config['op_keys'], config['red_op_keys'], num_classes=num_classes)
    
    @staticmethod
    def mnist_net(config, num_classes):
        return MNISTNet()
    
    @staticmethod
    def two_layer_net(config, num_classes):
        return TwoLayerNet(num_classes=num_classes)

# --
# Training helpers

class GridPointWorker(object):
    
    def __init__(self, config, net_class='rnet', dataset='CIFAR10', epochs=20, 
        lr_schedule='linear', lr_init=0.1, lr_fail_factor=0.5, max_failures=3, 
        cuda=True, dataset_num_workers=2):
    
        self.config = config
        self.config['vars'] = {
            "net_class"      : net_class,
            "dataset"        : dataset,
            "epochs"         : epochs,
            "lr_schedule"    : lr_schedule,
            "lr_init"        : lr_init,
            "lr_fail_factor" : lr_fail_factor,
            "max_failures"   : max_failures,
        }
        
        self.net_class      = net_class
        self.dataset        = dataset
        self.epochs         = epochs
        self.lr_schedule    = lr_schedule
        self.lr_init        = lr_init
        self.lr_fail_factor = lr_fail_factor
        self.max_failures   = max_failures
        self.cuda           = cuda
        
        self.fail_counter = 0
        self.hist = []
    
    def _reset_network(self):
        # Reset epoch count
        self.epoch = 0
        
        # Set dataset
        self.ds = getattr(DataLoader(root='../data/', pin_memory=self.cuda, 
            num_workers=dataset_num_workers), self.dataset)()
        
        # Network
        self.net = getattr(NetConstructors, self.net_class)(self.config, self.ds['num_classes'])
        self.net = self.net.cuda() if self.cuda else self.net
        
        # LR scheduler
        self.lr_scheduler = functools.partial(getattr(LRSchedule, self.lr_schedule), 
            lr_init=self.lr_init, epochs=self.epochs)
        
        # Optimizer -- need to allow configuration
        self.opt = torch.optim.SGD(self.net.parameters(), lr=self.lr_scheduler(float(self.epoch)), 
            momentum=0.9, weight_decay=5e-4)
    
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
            
            # Train for an epoch
            train_acc, train_loss, train_history = GridPointWorker._train_epoch(
                net=self.net,
                loader=self.ds['train_loader'],
                opt=self.opt,
                epoch=self.epoch,
                lr_scheduler=self.lr_scheduler,
                n_train_batches=self.ds['n_train_batches'],
            )
            
            # Sometimes the models don't converge...
            if np.isnan(train_loss):
                self.fail_counter += 1
                if self.fail_counter <= self.max_failures:
                    print >> sys.stderr, 'grid-point.py: train_loss is NaN -- reducing LR and restarting'
                    self.lr_init *= self.lr_fail_factor
                    self._reset_network()
                    continue
                else:
                    print >> sys.stderr, 'grid-point.py: train_loss is NaN -- too many failures -- exiting'
                    os._exit(0)
            
            # Eval on val + test datasets
            val_acc, val_loss = GridPointWorker._eval(self.net, self.epoch, self.ds['val_loader'], mode='val')
            test_acc, test_loss = GridPointWorker._eval(self.net, self.epoch, self.ds['test_loader'], mode='test')
            
            # Log performance
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

