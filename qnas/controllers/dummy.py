#!/usr/bin/env python

"""
    controller/dummy.py
"""

import sys
import numpy as np

class DummyController(object):
    
    def seed(self):
        return self._next()
    
    def success(self, last):
        config, hist = last
        print(config, file=sys.stderr)
    
    def failure(self, last):
        pass
    
    def _next(self):
        return {
            "func" : "run_dummy",
            "config" : {
                "dummy" : np.random.uniform()
            }
        }

