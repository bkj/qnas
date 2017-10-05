#!/usr/bin/env python

"""
    rq_functions.py
"""

from __future__ import print_function

import os
import time

from qnas.trainer import QNASTrainer
from qnas.controllers.dummy import DummyController
from qnas.controllers.optimizer_search import RandomOptimizerController

# --
# Function definitions

def kill(delay=1):
    print('!!! Kill Signal Received -- shutting down in %ds' % delay, file=sys.stderr)
    time.sleep(delay)
    os.kill(os.getppid(), 9)
    
def run_dummy(config, **kwargs):
    time.sleep(3)
    return config, {"dummy" : True}

def qnas_trainer_(config, **kwargs):
    qtrainer = QNASTrainer(config, **kwargs)
    results = qtrainer._train()
    # qtrainer.save()
    return qtrainer.config, results

# --
# Register possible functions that the workers can run

RQFunctions = {
    'run_dummy' : run_dummy,
    'kill' : kill,
    'qnas_trainer' : qnas_trainer_,
}

# --
# Register possible controllers

QNASControllers = {
    "DummyController" : DummyController,
    "RandomOptimizerController" : RandomOptimizerController,
}