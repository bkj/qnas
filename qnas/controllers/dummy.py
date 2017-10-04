#!/usr/bin/env python

"""
    controller/dummy.py
"""

import numpy as np

class DummyController(object):
    def next(self, last=None):
        if not last:
            return {
                "func" : "run_dummy",
                "config" : {
                    "dummy" : np.random.uniform()
                }
            }
        else:
            config, hist = last
            print config

