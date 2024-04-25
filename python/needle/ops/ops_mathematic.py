"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND 
from .ops_tuple import *

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        if BACKEND == "np":
            return array_api.add(a, self.scalar, dtype=a.dtype)
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        if BACKEND == "np":
            return array_api.multiply(a, self.scalar, dtype=a.dtype)
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a**b

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], NDArray) or not isinstance(
            node.inputs[1], NDArray
        ):
            raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * log(a)
        return grad_a, grad_b

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        if BACKEND == "np":
            return array_api.power(a, self.scalar, dtype=a.dtype)
        return a ** self.scalar

    def gradient(self, out_grad, node):
        return out_grad * self.scalar*(node.inputs[0]**(self.scalar-1))


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return a / b

    def gradient(self, out_grad, node):
        numerator, denominator = node.inputs
        return out_grad / denominator, -out_grad * numerator / (denominator ** 2)


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        if BACKEND == "np":
            return array_api.divide(a, self.scalar, dtype=a.dtype)
        return a / self.scalar

    def gradient(self, out_grad, node):
        return out_grad / self.scalar


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        if BACKEND == "np":
            if self.axes is None:
                return array_api.swapaxes(a, -1, -2)
            else:
                return array_api.swapaxes(a, self.axes[0], self.axes[1])

        dims = list(range(a.ndim))
        if self.axes is None:
            dims[-1], dims[-2] = dims[-2], dims[-1]
        else:
            dims[self.axes[0]], dims[self.axes[1]] = dims[self.axes[1]], dims[self.axes[0]]
        return a.permute(dims)
            

    def gradient(self, out_grad, node):
        return transpose(out_grad, self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.reshape(a.compact(), self.shape)

    def gradient(self, out_grad, node):
        return reshape(out_grad, node.inputs[0].shape)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        extra_dims = len(self.shape) - len(node.inputs[0].shape)
        in_dim = (1,)*extra_dims + node.inputs[0].shape
        axes = [i for i in range(len(self.shape)) if self.shape[i] != in_dim[i]]
        return reshape(summation(out_grad, axes=tuple(axes)), node.inputs[0].shape)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        axs = self.axes if isinstance(self.axes, tuple) else (self.axes,)
        for ax in sorted(axs, reverse=True):
            a = array_api.sum(a, axis=ax)
        return a

    def gradient(self, out_grad, node):
        dims = out_grad.shape
        if self.axes is not None:
            axs = self.axes if isinstance(self.axes, tuple) else (self.axes,)
            for ax in axs:
                dims = dims[:ax] + (1,) + dims[ax:]
        else:
            dims = (1,) * len(node.inputs[0].shape)
        return broadcast_to(reshape(out_grad, dims), node.inputs[0].shape)


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return a @ b

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        lhs_grad = matmul(out_grad, transpose(rhs))
        rhs_grad = matmul(transpose(lhs), out_grad)
        if lhs_grad.shape != lhs.shape:
            lhs_grad = summation(lhs_grad, axes=tuple(range(len(lhs_grad.shape) - len(lhs.shape))))
        if rhs_grad.shape != rhs.shape:
            rhs_grad = summation(rhs_grad, axes=tuple(range(len(rhs_grad.shape) - len(rhs.shape))))
        return lhs_grad, rhs_grad


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return -a

    def gradient(self, out_grad, node):
        return -out_grad


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        return out_grad / node.inputs[0]


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        return out_grad * exp(node.inputs[0])


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        return array_api.maximum(a, 0)

    def gradient(self, out_grad, node):
        return out_grad * (node.inputs[0].realize_cached_data() > 0)


def relu(a):
    return ReLU()(a)

class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * (-tanh(node.inputs[0])**2 + 1)
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        shape = args[0].shape
        for i in range(1, len(args)):
            assert args[i].shape == shape
        
        stacked_shape = list(shape)
        stacked_shape.insert(self.axis, len(args))
        stacked_arr = array_api.empty(stacked_shape, dtype=args[0].dtype, device=args[0].device)
        for i in range(len(args)):
            slices = [slice(None)]*len(stacked_shape)
            slices[self.axis] = i
            stacked_arr[tuple(slices)] = args[i]

        return stacked_arr
        ### END YOUR SOLUTION


    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        split_shape = list(A.shape)
        split_shape.pop(self.axis)
        split_arrs = [None]*A.shape[self.axis]
        for i in range(len(split_arrs)):
            new_arr = array_api.empty(split_shape, dtype=A.dtype, device=A.device)
            slices = [slice(None)]*len(A.shape)
            slices[self.axis] = i
            new_arr[tuple([slice(None)]*len(split_shape))] = A[tuple(slices)]
            split_arrs[i] = new_arr
        return tuple(split_arrs)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.flip(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        new_shape = list(a.shape)
        slices = [slice(None)]*len(new_shape)
        for ax in self.axes:
            if 0 <= ax < len(new_shape):
                new_shape[ax] *= self.dilation+1
                slices[ax] = slice(None, None, self.dilation+1)
        new_arr = array_api.full(new_shape, 0, dtype=a.dtype, device=a.device)
        new_arr[tuple(slices)] = a
        return new_arr
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        new_shape = list(a.shape)
        slices = [slice(None)]*len(new_shape)
        for ax in self.axes:
            if 0 <= ax < len(new_shape):
                new_shape[ax] //= self.dilation+1
                slices[ax] = slice(None, None, self.dilation+1)
        new_arr = array_api.empty(new_shape, dtype=a.dtype, device=a.device)
        new_arr = a[tuple(slices)]
        return new_arr
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        p = self.padding
        Z = A.pad(((0,0), (p,p), (p,p), (0,0)))
        N, H, W, C_in = Z.shape
        Ns,Hs,Ws,Cs = Z.strides

        KH,KW,KC,C_out = B.shape
        assert KC == C_in
        self.k = KH
        inner_dim = KH * KW * C_in
        weight = B.compact().reshape((inner_dim, C_out))

        s = self.stride
        out_h = (H-KH) // s + 1
        out_w = (W-KW) // s + 1

        return (Z.as_strided(shape  =(N , out_h, out_w, KH, KW, C_in), 
                             strides=(Ns, Hs*s , Ws*s , Hs, Ws, Cs)) \
                 .compact() \
                 .reshape((N*out_h*out_w, inner_dim)) \
                @ weight).reshape((N, out_h, out_w, C_out))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        A_grad = conv(
            dilate(
                out_grad, # (N, (H+2p-Kh)//s+1, (W+2p-Kw)//s+1, C_out)
                axes=(1,2),
                dilation=self.stride-1
            ), # (N, H+2p-Kh + 1, W+2p-Kw + 1, C_out)
            transpose(
                flip(
                    node.inputs[1], 
                    axes=(0,1)
                ) # flip kernel both ways
            ), # (Kh, Kw, C_out, C_in)
            stride=1,
            padding=self.k-1 - self.padding # 2Kh-2-2p, 2Kw-2-2p
        ) # 

        B_grad = transpose(
            transpose(
                conv(
                    transpose(
                        node.inputs[0], # (N, H, W, C_in)
                        axes=(0,3)
                    ), # (C_in, H, W, N)
                    transpose(
                        transpose(
                            dilate(
                                out_grad, # (N, (H+2p-Kh)//s+1, (W+2p-Kw)//s+1, C_out)
                                axes=(1,2),
                                dilation=self.stride-1
                            ), # (N, H+2p-Kh + 1, W+2p-Kw + 1, C_out)
                            axes=(0,1)
                        ), # (H+2p-Kh + 1, N, W+2p-Kw + 1, C_out)
                        axes=(1,2)
                    ), # (H+2p-Kh + 1, W+2p-Kw + 1, N, C_out)
                    stride=1,
                    padding=self.padding
                ), # (C_in, Kh, Kw, C_out)
                axes=(0,1)
            ),  # (Kh, C_in, Kw, C_out)
            axes=(1,2)
        ) # (Kh, Kw, C_in, C_out)
        return A_grad, B_grad
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)
