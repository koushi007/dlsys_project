from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        if BACKEND == "np":
            maxes = array_api.max(Z, axis=self.axes, keepdims=True)
            return array_api.log(array_api.sum(array_api.exp(Z - maxes), axis=self.axes)) + array_api.squeeze(maxes)

        if self.axes is not None:
            axs = self.axes if isinstance(self.axes, tuple) else (self.axes,)
            for ax in axs:
                shape = Z.shape[:ax] + (1,) + Z.shape[ax+1:]
        else:
            shape = (1,) * Z.ndim

        maxes = Z.max(axis=self.axes)
        reduced = Z - array_api.broadcast_to(maxes.reshape(shape), Z.shape)
        totals = array_api.sum(reduced.exp(), axis=self.axes)
        return totals.log()+maxes

    def gradient(self, out_grad, node):
        Z = node.inputs[0]

        dims = out_grad.shape
        if self.axes is not None:
            axs = self.axes if isinstance(self.axes, tuple) else (self.axes,)
            for ax in axs:
                dims = dims[:ax] + (1,) + dims[ax:]
        else:
            dims = (1,) * Z.ndim
        
        t = logsumexp(Z, axes=self.axes)
        return broadcast_to(reshape(out_grad, dims), Z.shape) \
            * exp(Z - broadcast_to(reshape(t, dims), Z.shape))


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

