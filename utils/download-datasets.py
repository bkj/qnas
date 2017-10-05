#!/usr/bin/env python

"""
    download-datasets.py
    
    - Download (some of) the `torchvision` datasets
    - Convert to directories of images
        - That way, can use `ImageFolder` dataloader for all datasets, do manual validation splits, etc
"""

from __future__ import print_function

import os
import sys
import argparse
from tqdm import tqdm
from torchvision import datasets

# --
# Helpers

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default='MNIST,fashionMNIST,STL10,CIFAR10,CIFAR100,SVHN')
    parser.add_argument('--root', type=str, default='./data/')
    return parser.parse_args()

class Downloader(object):
    
    @staticmethod
    def _download(root, name, dataset, mode):
        for i, (img, lab) in tqdm(enumerate(dataset), total=len(dataset)):
            outdir = os.path.join(root, 'manual', name, mode, str(lab))
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            
            img.save(os.path.join(outdir, '%d.jpg' % i))
    
    @staticmethod
    def MNIST(root):
        name = 'MNIST'
        Downloader._download(root, name, datasets.MNIST(os.path.join(root, name), download=True, train=True), 'train')
        Downloader._download(root, name, datasets.MNIST(os.path.join(root, name), download=True, train=False), 'test')
    
    @staticmethod
    def fashionMNIST(root):
        name = 'fashionMNIST'
        Downloader._download(root, name, datasets.FashionMNIST(os.path.join(root, name), download=True, train=True), 'train')
        Downloader._download(root, name, datasets.FashionMNIST(os.path.join(root, name), download=True, train=False), 'test')
    
    @staticmethod
    def STL10(root):
        name = 'STL10'
        Downloader._download(root, name, datasets.STL10(os.path.join(root, name), download=True, split='train'), 'train')
        Downloader._download(root, name, datasets.STL10(os.path.join(root, name), download=True, split='test'), 'test')
    
    @staticmethod
    def CIFAR10(root):
        name = 'CIFAR10'
        Downloader._download(root, name, datasets.CIFAR10(os.path.join(root, name), download=True, train=True), 'train')
        Downloader._download(root, name, datasets.CIFAR10(os.path.join(root, name), download=True, train=False), 'test')
    
    @staticmethod
    def CIFAR100(root):
        name = 'CIFAR100'
        Downloader._download(root, name, datasets.CIFAR100(os.path.join(root, name), download=True, train=True), 'train')
        Downloader._download(root, name, datasets.CIFAR100(os.path.join(root, name), download=True, train=False), 'test')
    
    @staticmethod
    def SVHN(root):
        name = 'SVHN'
        Downloader._download(root, name, datasets.SVHN(os.path.join(root, name), download=True, split='train'), 'train')
        Downloader._download(root, name, datasets.SVHN(os.path.join(root, name), download=True, split='test'), 'test')


if __name__ == "__main__":
    args = parse_args()
    
    for name in list(set(args.datasets.split(','))):
        print('downloading %s' % name, file=sys.stderr)
        getattr(Downloader, name)(args.root)

