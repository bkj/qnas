#!/usr/bin/env python

"""
    data.py
    
    Data loaders for NAS
"""

import os
import torch
import torchvision
from torchvision import transforms, datasets

class DataLoader(object):
    
    def __init__(self, root='./data/', num_workers=8, pin_memory=True):
        self.root = root
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def _basic(self, name, center=False, hflip=False):
        
        transform_train = []
        transform_test = []
        
        if hflip:
            transform_train.append(transforms.RandomHorizontalFlip())
        
        transform_train.append(transforms.ToTensor())
        transform_test.append(transforms.ToTensor())
        
        if center:
            transform_train.append(transforms.Lambda(lambda x: x - x.mean()))
            transform_test.append(transforms.Lambda(lambda x: x - x.mean()))
        
        if name == 'MNIST':
            transform_train.append(transforms.Lambda(lambda x: x[0:1,:,:]))
            transform_test.append(transforms.Lambda(lambda x: x[0:1,:,:]))
        
        transform_train = transforms.Compose(transform_train)
        transform_test = transforms.Compose(transform_test)
        
        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(root=os.path.join(self.root, 'manual/%s/tv_train' % name), transform=transform_train), 
            batch_size=128, 
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(root=os.path.join(self.root, 'manual/%s/tv_val' % name), transform=transform_test), 
            batch_size=256, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        
        test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(root=os.path.join(self.root, 'manual/%s/test' % name), transform=transform_test), 
            batch_size=256, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        
        n_train_batches = len(train_loader)
        n_val_batches = len(val_loader)
        n_test_batches = len(test_loader)
        
        return {
            'train_loader'    : train_loader, 
            'val_loader'      : val_loader,
            'test_loader'     : test_loader, 
            'n_train_batches' : n_train_batches,
            'n_val_batches'   : n_val_batches,
            'n_test_batches'  : n_test_batches,
            'num_classes'     : len(train_loader.dataset.classes),
        }
        
    def MNIST():
        return self._basic(name='MNIST', center=False, hflip=False)
        
    def fashionMNIST():
        return self._basic(name='fashionMNIST', center=False, hflip=True)
        
    def SVHN():
        return self._basic(name='SVHN', center=True, hflip=False)
        
    def STL10():
        return self._basic(name='STL10', center=True, hflip=True)
    
    def _CIFAR(name='CIFAR10'):
        
        mn = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*mn),
        ])
            
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*mn),
        ])
        
        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(root=os.path.join(self.root, 'manual/%s/tv_train' % name), transform=transform_train), 
            batch_size=128, 
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(root=os.path.join(self.root, 'manual/%s/tv_val' % name), transform=transform_test), 
            batch_size=256, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        
        test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(root=os.path.join(self.root, 'manual/%s/test' % name), transform=transform_test), 
            batch_size=256, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        
        n_train_batches = len(train_loader)
        n_val_batches = len(val_loader)
        n_test_batches = len(test_loader)
        
        return {
            'train_loader'    : train_loader, 
            'val_loader'      : val_loader,
            'test_loader'     : test_loader, 
            'n_train_batches' : n_train_batches,
            'n_val_batches'   : n_val_batches,
            'n_test_batches'  : n_test_batches,
            'num_classes'     : len(train_loader.dataset.classes),
        }
    
    def CIFAR10():
        return self._CIFAR(name='CIFAR10')
    
    def CIFAR100():
        return self._CIFAR(name='CIFAR100')
