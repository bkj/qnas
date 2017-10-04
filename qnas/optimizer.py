#!/usr/bin/env python

"""
    optimizer.py
    
    *_space arrays give operands/operations as defined in the paper
"""

from functools import partial

import torch
from uuid import uuid4
from torch.autograd import Variable
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer


class Operands(object):
    
    stateful = set([
        'grad_expavg',
        'grad_expavg_sign',
    ])
    
    @staticmethod
    def grad_power(grad, state, params):
        power = params
        return grad ** power, state
    
    @staticmethod
    def grad_expavg(grad, state, params):
        power, beta, mid = params
        k = 'grad_expavg_%s' % mid
        
        if k not in state:
            state[k] = (grad ** power).clone()
        else:
            state[k].mul_(beta).add_((1 - beta) * grad ** power)
        
        return state[k], state
    
    @staticmethod
    def grad_sign(grad, state, params):
        power = params
        return torch.sign(grad ** power), state
    
    @staticmethod
    def grad_expavg_sign(grad, state, params):
        x, state = Operands.grad_expavg(grad, state, params)
        return torch.sign(x), state
    
    @staticmethod
    def const(grad, state, params):
        const = params
        const = torch.FloatTensor([const])
        if grad.is_cuda:
            const = const.cuda()
        
        return const, state
    
    @staticmethod
    def gaussian_noise(grad, state, params):
        mn, sd = params
        
        noise = torch.randn(grad.size())
        if grad.is_cuda:
            noise = noise.cuda()
        
        return mn + sd * noise, state
    
    @staticmethod
    def compound(grad, state, params):
        return parse_arch(params)(grad, state)


class UnaryOperation(object):
    
    @staticmethod
    def identity(x, params):
        return x
    
    @staticmethod
    def negation(x, params):
        return -x
    
    @staticmethod
    def exp(x, params):
        return torch.exp(x)
    
    @staticmethod
    def abs_log(x, params):
        return torch.log(torch.abs(x))
    
    @staticmethod
    def abs_pow(x, params):
        power = params
        return torch.pow(torch.abs(x), power)
    
    @staticmethod
    def clip(x, params):
        bound = params
        return torch.clamp(x, -bound, bound)
    
    @staticmethod
    def drop(x, params):
        prob = params
        return F.dropout(x, prob, training=True).data
    
    @staticmethod
    def sign(x, params):
        return torch.sign(x)


class BinaryOperation(object):
    @staticmethod
    def add(x, y, params):
        return x + y
    
    @staticmethod
    def avg(x, y, params):
        return (x + y) / 2
    
    @staticmethod
    def sub(x, y, params):
        return x - y
    
    @staticmethod
    def mul(x, y, params):
        return x * y
    
    @staticmethod
    def div(x, y, params):
        delta = params
        return x / (y + delta)
    
    # @staticmethod
    # def exp(x, y, params):
    #     # !! negative number cannot be raised to a fractional power... so not sure what to do
    #     return grad_a ** grad_b
        
    @staticmethod
    def keep_left(x, y, params):
        return x


def parse_arch(arch):
    
    for k,v in arch.items():
        if isinstance(v, str):
            arch[k] = (v, None)
        elif v[0] in Operands.stateful:
            # add unique ID if Operand is stateful (eg to avoid collisions)
            arch[k] = (v[0], v[1][:-1] + (str(uuid4()),))
    
    op1 = partial(getattr(Operands, arch['op1'][0]), params=arch['op1'][1])
    un1 = partial(getattr(UnaryOperation, arch['un1'][0]), params=arch['un1'][1])
    
    op2 = partial(getattr(Operands, arch['op2'][0]), params=arch['op2'][1])
    un2 = partial(getattr(UnaryOperation, arch['un2'][0]), params=arch['un2'][1])
    
    bin_ = partial(getattr(BinaryOperation, arch['bin'][0]), params=arch['bin'][1])
    
    def f(grad, state):
        left, state = op1(grad, state)
        right, state = op2(grad, state)
        return bin_(un1(left), un2(right)), state
    
    return f


class ConfigurableOptimizer(Optimizer):
    def __init__(self, params, arch, lr=0.1):
        self.update_rule = parse_arch(arch)
        super(ConfigurableOptimizer, self).__init__(params, {
            "lr" : lr,
        })
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                update, self.state[p] = self.update_rule(grad, self.state[p])
                p.data.sub_(group['lr'], update)
        
        return loss
