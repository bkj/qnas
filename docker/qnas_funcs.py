#!/usr/bin/env python

"""
    qnas_funcs.py
"""

import os
import sys
import time
from datetime import datetime

def run_job(x):
    time.sleep(2)
    return x, x ** 2

def kill():
    print >> sys.stderr, 'kill signal received at %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    os._exit(0)