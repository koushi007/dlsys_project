"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        for param in self.params:
            grad = param.grad.data
            
            if self.weight_decay != 0.0:
                grad.data = grad.data + self.weight_decay*param.data
                
            if self.momentum != 0.0:
                if param not in self.u:
                    self.u[param] = ndl.zeros_like(param.data).data

                self.u[param].data = self.momentum*self.u[param].data - (1.0-self.momentum)*grad.data
                param.data = param.data + self.lr*self.u[param].data

            else:
                param.data = param.data - self.lr*grad.data

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        self.t += 1
        
        for param in self.params:
            grad = param.grad.data
            
            if self.weight_decay != 0.0:
                grad.data = grad.data + self.weight_decay*param.data

            if param not in self.m:
                self.m[param] = ndl.zeros_like(param.data).data
            if param not in self.v:
                self.v[param] = ndl.zeros_like(param.data).data
            
            self.m[param].data = self.beta1*self.m[param].data + (1.0-self.beta1)*grad.data
            self.v[param].data = self.beta2*self.v[param].data + (1.0-self.beta2)*grad.data**2.0

            m_hat = self.m[param].data / (1.0-self.beta1**self.t)
            v_hat = self.v[param].data / (1.0-self.beta2**self.t)

            param.data = param.data - self.lr*m_hat / (v_hat**0.5 + self.eps)
