#!/usr/bin/env python

"""
    rq_functions.py
"""

import os
import sys
import time

sys.path.append('../qnas')
from qnas_trainer import QNASTrainer
from controllers.dummy import DummyController
from controllers.optimizer_search import RandomOptimizerController

# --
# Function definitions

def kill(delay=1):
    print >> sys.stderr, '!!! Kill Signal Received -- shutting down in %ds' % delay
    time.sleep(delay)
    os.kill(os.getppid(), 9)
    
def run_dummy(config, **kwargs):
    time.sleep(3)
    return config, {"dummy" : True}

def qnas_trainer_(config, **kwargs):
    qtrainer = QNASTrainer(config, **kwargs)
    results = qtrainer._train()
    qtrainer.save()
    return qtrainer.config, results

# --
# Dict for RQMaster

RQFunctions = {
    'run_dummy' : run_dummy,
    'kill' : kill,
    'qnas_trainer' : qnas_trainer_,
}

QNASControllers = {
    "DummyController" : DummyController,
    "RandomOptimizerController" : RandomOptimizerController,
}