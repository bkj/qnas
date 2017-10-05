#!/usr/bin/env python

"""
    controller/dummy.py
"""

import numpy as np

class DummyController(object):
    
    def seed(self):
        return self._next()
    
    def success(self, last):
        config, hist = last
        print config
    
    def failure(self, last):
        pass
    
    def _next(self):
        return {
            "func" : "run_dummy",
            "config" : {
                "dummy" : np.random.uniform()
            }
        }

