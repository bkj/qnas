#!/usr/bin/env python

"""
    controllers/dummy.py
"""

from __future__ import print_function

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

