#!/usr/bin/env python

"""
    rq_functions.py
"""

from __future__ import print_function

import os
import sys

if sys.version_info.major < 3:
    raise Exception('sys.version_info.major < 3')
    os._exit(1)

import time
from uuid import getnode as get_mac

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
    results = qtrainer.train()
    # qtrainer.save()
    config = qtrainer.config
    config.update({
        "_meta" : get_mac()
    })

# --
# Register possible functions for `rqworkers` to run

RQFunctions = {
    'run_dummy' : run_dummy,
    'kill' : kill,
    'qnas_trainer' : qnas_trainer_,
}

# --
# Register possible controllers for `rq_master.py` to run

QNASControllers = {
    "DummyController" : DummyController,
    "RandomOptimizerController" : RandomOptimizerController,
}
