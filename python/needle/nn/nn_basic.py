"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
import math


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype))
        if bias:
            self.bias = init.kaiming_uniform(out_features, 1, device=device, dtype=dtype).transpose().detach()
            self.bias.requires_grad = True
            self.bias = Parameter(self.bias)
        else:
            self.bias = None

    def forward(self, X: Tensor) -> Tensor:
        Z = X.matmul(self.weight)
        if self.bias:
            return Z + ops.broadcast_to(self.bias, Z.shape)
        return Z


class Flatten(Module):
    def forward(self, X):
        return X.reshape((X.shape[0], math.prod(X.shape[1:])))


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        Z = x
        for f in self.modules:
            Z = f(Z)
        return Z


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        y_one_hot = init.one_hot(logits.shape[-1], y, device=logits.device, dtype=logits.dtype).data
        return ops.summation(ops.logsumexp(logits, axes=(1,)) - ops.summation(logits*y_one_hot, axes=(1,))) / logits.shape[0]


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype).data
        self.running_var = init.ones(dim, device=device, dtype=dtype).data

    def forward(self, x: Tensor) -> Tensor:
        weights_bc = ops.broadcast_to(ops.reshape(self.weight, (1, self.dim)), x.shape)
        bias_bc = ops.broadcast_to(ops.reshape(self.bias, (1, self.dim)), x.shape)

        if self.training:
            means = ops.summation(x, axes=(0,)) / x.shape[0]
            self.running_mean.data = (1.0 - self.momentum)*self.running_mean.data + self.momentum*means.data
            means_kd = means.reshape((1, x.shape[1])).broadcast_to(x.shape)
            mean_diffs = x - means_kd
            
            vars = (ops.summation((mean_diffs)**2.0, axes=(0,)) / x.shape[0])
            self.running_var.data = (1.0 - self.momentum)*self.running_var.data + self.momentum*vars.data
            vars_kd = vars.reshape((1, x.shape[1])).broadcast_to(x.shape)
            
        else:
            means_kd = self.running_mean.reshape((1, x.shape[1])).broadcast_to(x.shape)
            mean_diffs = x - means_kd

            vars_kd = self.running_var.reshape((1, x.shape[1])).broadcast_to(x.shape)

        return weights_bc*(mean_diffs / (vars_kd + self.eps)**0.5) + bias_bc

class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        means = ops.summation(x, axes=(1,)) / x.shape[1]
        means_kd = means.reshape((x.shape[0], 1)).broadcast_to(x.shape)
        mean_diffs = x - means_kd
        
        vars = (ops.summation((mean_diffs)**2.0, axes=(1,)) / x.shape[1])
        vars_kd = vars.reshape((x.shape[0], 1)).broadcast_to(x.shape)
        
        weights_bc = self.weight.reshape((1, self.dim)).broadcast_to(x.shape)
        bias_bc = self.bias.reshape((1, self.dim)).broadcast_to(x.shape)
        
        return  weights_bc*(mean_diffs / (vars_kd + self.eps)**0.5) + bias_bc


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            return x * init.randb(*x.shape, p=1.0-self.p, device=x.device, dtype=x.dtype) / (1.0-self.p)
        return x


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x) + x
