#!/usr/bin/env python

"""
    optnet.py
    
    Wrapper around TwoLayerNet that allows for custom optimizer
"""

sys.path.append('..')
from two_layer_net import TwoLayerNet
from optimizer import ConfigurableOptimizer

class OptNetSmall(TwoLayerNet):
    def __init__(self, config, **kwargs):
        super(OptNetSmall, self).__init__(config, **kwargs)
        self.opt = ConfigurableOptimizer(self.parameters(), config['opt_arch'])