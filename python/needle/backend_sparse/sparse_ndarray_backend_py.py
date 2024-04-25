from typing import Callable
from collections import deque, defaultdict

import numpy as np
import itertools

__device_name__ = "numpy"
_datatype = np.float32
_datetype_size = np.dtype(_datatype).itemsize


####################
### SPARSE ARRAY ###
####################
class SparseArray:
    def __init__(self, 
                 shape: tuple[int, ...], 
                 default_val: float = 0, 
                 indices: np.ndarray = None,
                 values: np.ndarray = None):
        self.shape = shape
        self.ndim = len(shape)
        self.default_val = default_val

        if indices is None:
            self.indices = np.empty((0, self.ndim), dtype=np.uint32)
        else:
            self.indices = indices

        if values is None:
            self.values = np.empty((0,), dtype=_datatype)
        else:
            self.values = values
        
        SparseArray.check(shape, self.indices, self.values)

        self.size = self.values.size

    def fill(self, value):
        self.default_val = value
        self.indices = np.empty((0, self.ndim), dtype=np.uint32)
        self.values = np.empty((0,), dtype=_datatype)
        self.size = 0

    @staticmethod
    def check(shape, indices, values):
        assert isinstance(shape, tuple)
        assert indices.dtype == np.uint32
        assert values.dtype == _datatype
        assert values.ndim == 1
        assert indices.shape[0] == values.shape[0]
        assert indices.shape[1] == len(shape)


#################
### UTILITIES ###
#################
def from_numpy(shape, default_val, indices, values, out: SparseArray):
    shape = tuple(shape)
    values = np.array(values, dtype=_datatype)
    indices = np.array(indices, dtype=np.uint32).reshape((len(values), len(shape)))
    SparseArray.check(shape, indices, values)

    out.shape = shape
    out.ndim = len(shape)
    out.default_val = default_val
    out.indices = indices
    out.values = values
    out.size = values.size


def compact_strides(shape: tuple[int, ...]):
        """Utility function to compute compact strides"""
        stride = 1
        res = []
        for i in range(1, len(shape) + 1):
            res.append(stride)
            stride *= shape[-i]
        return tuple(res[::-1])


##############################
### ARRAY MANIPULATION OPS ###
##############################
def reshape(a: SparseArray, new_shape: tuple[int, ...], out: SparseArray):
    out.default_val = a.default_val
    out.shape = new_shape
    out.ndim = len(new_shape)
    out.values = a.values.copy()
    out.size = a.size

    out.indices = np.sum(a.indices * compact_strides(a.shape), axis=1)
    out.indices = np.array(np.unravel_index(out.indices, new_shape)).T
    out.indices = out.indices.astype(np.uint32)

def permute(a: SparseArray, new_axes: tuple[int, ...], out: SparseArray):
    out.default_val = a.default_val
    out.ndim = len(out.shape)
    out.indices = a.indices[:, new_axes]
    out.values = a.values.copy()

    order = np.argsort((out.indices * compact_strides(out.shape)).sum(axis=1))

    out.indices = out.indices[order]
    out.values = out.values[order]
    out.size = out.values.size

def broadcast_to(a: SparseArray, new_shape: tuple[int, ...], out: SparseArray):
    out.default_val = a.default_val
    out.shape = new_shape
    out.ndim = len(new_shape)
    out.indices = a.indices.copy()
    out.values = a.values.copy()
    out.size = a.size
    
    for ax, (new_ax, old_ax) in enumerate(zip(new_shape[::-1], a.shape[::-1])):
        if new_ax == old_ax:
            continue
        elif old_ax == 1:
            out.indices = np.tile(out.indices, (new_ax, 1))
            for i in range(out.size):
                out.indices[i*out.size:(i+1)*out.size, out.ndim-1-ax] = i
            out.values = np.tile(out.values, new_ax)
            out.size = out.values.size
        else:
            raise RuntimeError("Cannot broadcast to shape")

##################
### GETITEM OP ###
##################
def getitem(a: SparseArray, idxs: tuple[tuple, ...], out: SparseArray):
    
    new_indices, new_values = a.indices.copy(), a.values.copy()

    mask = np.zeros_like(new_indices, dtype=np.bool_)
    for s_ind, s in enumerate(idxs):
        mask[:, s_ind] = (new_indices[:, s_ind][:, None] == 
                    np.arange(s[0], s[1], s[2])[None, :]).any(axis=1)
        new_indices[:, s_ind] = (new_indices[:, s_ind] - s[0]) / s[2]
    
    mask = mask.all(axis=1)
    new_indices = new_indices[mask]
    new_values = new_values[mask]

    # need to process each slice so it's formatted properly similar to HW
    # then loop until start index matched 
    out.default_val = a.default_val
    out.indices = new_indices
    out.values = new_values
    out.size = out.values.size


###################
### SETITEM OPS ###
###################
def scalar_setitem(a: SparseArray, idxs: tuple[slice, ...], val: float):
    if val == a.default_val:
        mask = np.zeros_like(a.indices, dtype=np.bool_)
        for s_ind, s in enumerate(idxs):
            mask[:, s_ind] = (a.indices[:, s_ind][:, None] == 
                        np.arange(s.start, s.stop, s.step)[None, :]).any(axis=1)

        mask = mask.all(axis=1)
        a.indices = a.indices[~mask]
        a.values = a.values[~mask]
        return

    a_idx = 0
    new_indices = deque()
    new_values = deque()

    for i in itertools.product(*[list(range(s.start, s.stop, s.step)) for s in idxs]):
        while a_idx < a.size and tuple(a.indices[a_idx]) < i:
            new_indices.append(tuple(a.indices[a_idx]))
            new_values.append(a.values[a_idx])
            a_idx += 1

        if a_idx < a.size and tuple(a.indices[a_idx]) == i:
            new_indices.append(i)
            new_values.append(val)
            a_idx += 1
        else:
            new_indices.append(i)
            new_values.append(val)

    while a_idx < a.size:
        new_indices.append(tuple(a.indices[a_idx]))
        new_values.append(a.values[a_idx])
        a_idx += 1

    if new_values:
        a.indices = np.array(new_indices, dtype=np.uint32)
        a.values = np.array(new_values, dtype=_datatype)
        a.size = a.values.size
    else:
        a.indices = np.empty((0, a.ndim), dtype=np.uint32)
        a.values = np.empty((0,), dtype=_datatype)
        a.size = 0


def ewise_setitem(a: SparseArray, idxs: tuple[slice, ...], b: SparseArray):

    a_idx = 0
    b_idx = 0
    new_indices = deque()
    new_values = deque()

    idxs_a = [list(range(s.start, s.stop, s.step)) for s in idxs]
    idxs_b = [list(range(0, dim)) for dim in b.shape]

    for i_a, i_b in zip(itertools.product(*idxs_a), itertools.product(*idxs_b)):
        while a_idx < a.size and tuple(a.indices[a_idx]) < i_a:
            new_indices.append(tuple(a.indices[a_idx]))
            new_values.append(a.values[a_idx])
            a_idx += 1

        if a_idx < a.size and tuple(a.indices[a_idx]) == i_a:
            if b_idx < b.size and i_b == tuple(b.indices[b_idx]):
                new_indices.append(i_a)
                new_values.append(b.values[b_idx])
                b_idx += 1
            else:
                if b.default_val != a.default_val:
                    new_indices.append(i_a)
                    new_values.append(b.default_val)
            a_idx += 1
        else:
            if b_idx < b.size and i_b == tuple(b.indices[b_idx]):
                new_indices.append(i_a)
                new_values.append(b.values[b_idx])
                b_idx += 1
            else:
                raise RuntimeError("Should never get here")

    while a_idx < a.size:
        new_indices.append(tuple(a.indices[a_idx]))
        new_values.append(a.values[a_idx])
        a_idx += 1

    if new_values:
        a.indices = np.array(new_indices, dtype=np.uint32)
        a.values = np.array(new_values, dtype=_datatype)
        a.size = a.values.size
    else:
        a.indices = np.empty((0, a.ndim), dtype=np.uint32)
        a.values = np.empty((0,), dtype=_datatype)
        a.size = 0
    
    
#################
### SCALAR OP ###
#################
def scalar_op(a: SparseArray, val: float, out: SparseArray, op: Callable, sparsify: bool = True):
    out.default_val = op(a.default_val, val)
    out.shape = a.shape
    out.ndim = a.ndim

    out.indices = a.indices.copy()
    out.values = op(a.values, val).astype(_datatype)
    out.size = out.values.size

    if sparsify:
        # Remove indices where the value is now the default value
        mask = out.values != out.default_val
        out.indices = out.indices[mask]
        out.values = out.values[mask]
        out.size = out.values.size
    

######################
### SINGLE ARR OPS ###
######################
def ewise_neg(a: SparseArray, out: SparseArray):
    scalar_op(a, 0, out, lambda x, _: -x)

def ewise_log(a: SparseArray, out: SparseArray):
    scalar_op(a, 0, out, lambda x, _: np.log(x))

def ewise_exp(a: SparseArray, out: SparseArray):
    scalar_op(a, 0, out, lambda x, _: np.exp(x))

def ewise_tanh(a: SparseArray, out: SparseArray):
    scalar_op(a, 0, out, lambda x, _: np.tanh(x))


########################
### SCALAR ARITH OPS ###
########################
def scalar_add(a: SparseArray, val: float, out: SparseArray):
    scalar_op(a, val, out, lambda x, y: x + y)

def scalar_sub(a: SparseArray, val: float, out: SparseArray):
    scalar_op(a, val, out, lambda x, y: x - y)

def scalar_mul(a: SparseArray, val: float, out: SparseArray):
    scalar_op(a, val, out, lambda x, y: x * y)

def scalar_div(a: SparseArray, val: float, out: SparseArray):
    scalar_op(a, val, out, lambda x, y: x / y)

def scalar_power(a: SparseArray, val: float, out: SparseArray):
    scalar_op(a, val, out, lambda x, y: x**y)


#######################
### SCALAR COMP OPS ###
#######################
def scalar_maximum(a: SparseArray, val: float, out: SparseArray):
    scalar_op(a, val, out, lambda x, y: np.maximum(x, y))

def scalar_eq(a: SparseArray, val: float, out: SparseArray):
    scalar_op(a, val, out, lambda x, y: (x == y))

def scalar_ne(a: SparseArray, val: float, out: SparseArray):
    scalar_op(a, val, out, lambda x, y: (x != y))

def scalar_ge(a: SparseArray, val: float, out: SparseArray):
    scalar_op(a, val, out, lambda x, y: (x >= y))

def scalar_le(a: SparseArray, val: float, out: SparseArray):
    scalar_op(a, val, out, lambda x, y: (x <= y))

def scalar_gt(a: SparseArray, val: float, out: SparseArray):
    scalar_op(a, val, out, lambda x, y: (x > y))

def scalar_lt(a: SparseArray, val: float, out: SparseArray):
    scalar_op(a, val, out, lambda x, y: (x < y))


################
### EWISE OP ###
################
def ewise_op(a: SparseArray, b: SparseArray, out: SparseArray, op: Callable, sparsify: bool = True):
    assert a.shape == b.shape
    assert a.ndim == b.ndim

    out.shape = a.shape
    out.ndim = a.ndim
    out.default_val = op(a.default_val, b.default_val)

    a_idx = 0
    b_idx = 0
    new_indices = deque()
    new_values = deque()

    while a_idx < a.size and b_idx < b.size:
        a_i = tuple(a.indices[a_idx])
        b_i = tuple(b.indices[b_idx])

        if a_i == b_i:
            new_val = op(a.values[a_idx], b.values[b_idx])
            if not sparsify or new_val != out.default_val:
                new_indices.append(a_i)
                new_values.append(new_val)
            a_idx += 1
            b_idx += 1
        elif a_i <= b_i:
            new_val = op(a.values[a_idx], b.default_val)
            if not sparsify or new_val != out.default_val:
                new_indices.append(a_i)
                new_values.append(new_val)
            a_idx += 1
        elif a_i >= b_i:
            new_val = op(a.default_val, b.values[b_idx])
            if not sparsify or new_val != out.default_val:
                new_indices.append(b_i)
                new_values.append(new_val)
            b_idx += 1
        else:
            raise RuntimeError("Should never get here")

    while a_idx < a.size:
        a_i = tuple(a.indices[a_idx])
        new_val = op(a.values[a_idx], b.default_val)
        if not sparsify or new_val != out.default_val:
            new_indices.append(a_i)
            new_values.append(new_val)
        a_idx += 1

    while b_idx < b.size:
        b_i = tuple(b.indices[b_idx])
        new_val = op(a.default_val, b.values[b_idx])
        if not sparsify or new_val != out.default_val:
            new_indices.append(b_i)
            new_values.append(new_val)
        b_idx += 1

    if new_values:
        out.indices = np.array(new_indices, dtype=np.uint32)
        out.values = np.array(new_values, dtype=_datatype)
        out.size = out.values.size
    else:
        out.indices = np.empty((0, a.ndim), dtype=np.uint32)
        out.values = np.empty((0,), dtype=_datatype)
        out.size = 0


#######################
### EWISE ARITH OPS ###
#######################
def ewise_add(a: SparseArray, b: SparseArray, out: SparseArray):
    ewise_op(a, b, out, lambda x, y: x + y)

def ewise_sub(a: SparseArray, b: SparseArray, out: SparseArray):
    ewise_op(a, b, out, lambda x, y: x - y)

def ewise_mul(a: SparseArray, b: SparseArray, out: SparseArray):
    ewise_op(a, b, out, lambda x, y: x * y)

def ewise_div(a: SparseArray, b: SparseArray, out: SparseArray):
    ewise_op(a, b, out, lambda x, y: x / y)


######################
### EWISE COMP OPS ###
######################
def ewise_maximum(a: SparseArray, b: SparseArray, out: SparseArray):
    ewise_op(a, b, out, lambda x, y: np.maximum(x, y))

def ewise_eq(a: SparseArray, b: SparseArray, out: SparseArray):
    ewise_op(a, b, out, lambda x, y: (x == y))

def ewise_ne(a: SparseArray, b: SparseArray, out: SparseArray):
    ewise_op(a, b, out, lambda x, y: (x != y))

def ewise_ge(a: SparseArray, b: SparseArray, out: SparseArray):
    ewise_op(a, b, out, lambda x, y: (x >= y))

def ewise_le(a: SparseArray, b: SparseArray, out: SparseArray):
    ewise_op(a, b, out, lambda x, y: (x <= y))

def ewise_gt(a: SparseArray, b: SparseArray, out: SparseArray):
    ewise_op(a, b, out, lambda x, y: (x > y))

def ewise_lt(a: SparseArray, b: SparseArray, out: SparseArray):
    ewise_op(a, b, out, lambda x, y: (x < y))


##############
### MATMUL ###
##############
def matmul(a: SparseArray, b: SparseArray, out: SparseArray):
    assert a.shape[-1] == b.shape[-2]
    out.shape = (a.shape[0],b.shape[1])
    out.ndim = 2
    out.default_val = a.default_val * b.default_val * a.shape[-1]
    ## a.indices -> (a_size,2)
    ## b.indices -> (b_size,2)
    ## a.values -> (a_size,)
    ## a_ind_0,a_ind_1 = a.indices[:,0],a.indices[:,1]
    ## b_ind_0,b_ind_1 = b.indices[:,0],b.indices[:,1]
    ## dict_a_ind_1 -> tuple(a_ind_0,a_val)
    ## dict_b_ind_0 -> tuple(b_ind_1,b_val)
    a_devs = a.values - a.default_val
    b_devs = b.values - b.default_val
    
    dict_a_ind_1 = defaultdict(lambda : [])
    dict_b_ind_0 = defaultdict(lambda : [])
    for i in range(a.size):
        dict_a_ind_1[a.indices[i][1]].append((a.indices[i][0], a_devs[i]))
    for i in range(b.size):
        dict_b_ind_0[b.indices[i][0]].append((b.indices[i][1], b_devs[i]))
    ## out_dict : dict of (i,j) -> value
    out_dict = defaultdict(lambda : 0)
    ## for each key in dict_a_ind_1 and matching key in dict_b_ind_0
    ## multiply values and add to out_dict
    for idx,a_list in dict_a_ind_1.items():
        b_list = dict_b_ind_0[idx]
        for i,a_val in a_list:
            for j,b_val in b_list:
                out_dict[(i,j)] += a_val * b_val

    if a.default_val != 0:
        for idx,val in zip(a.indices, a_devs):
            for i in range(b.shape[1]):
                out_dict[(idx[0],i)] += val * b.default_val

    if b.default_val != 0:
        for idx,val in zip(b.indices, b_devs):
            for i in range(a.shape[0]):
                out_dict[(i,idx[1])] += a.default_val * val

    ## convert out_dict to out.indices and out.values
    out_indices = deque([])
    out_values = deque([])
    for idx,val in sorted(out_dict.items()):
        out_indices.append(idx)
        out_values.append(val + out.default_val)
    out_indices = np.array(out_indices,dtype=np.uint32)
    out_values = np.array(out_values,dtype=_datatype)
    out.indices = out_indices
    out.values = out_values
    out.size = out_values.size



##################
### REDUCE OPS ###
##################
def reduce_sum(a: SparseArray, axis: int, keepdims: bool, out: SparseArray):
    reduced_shape = tuple([s for i, s in enumerate(a.shape) if i != axis])
    reduced_strides = compact_strides(reduced_shape)
    reduced_indices = np.delete(a.indices, axis, axis=1)
    order = np.argsort((reduced_indices * reduced_strides).sum(axis=1))
    unique, split_inds, counts = np.unique(reduced_indices[order], axis=0,
                                           return_index=True, return_counts=True)
    if keepdims:
        out.indices = np.insert(unique, axis, 0, axis=1)
    else:
        out.indices = unique

    splits = np.split(a.values[order], split_inds[1:])
    max_len = np.max(counts)
    splits = [np.pad(s, (0, max_len - counts[i])) for i, s in enumerate(splits)]
    out.values = np.sum(splits, axis=1) \
                    + a.default_val*(a.shape[axis] - counts)

    out.size = out.values.size
    out.default_val = a.default_val * a.shape[axis]
            

def reduce_max(a: SparseArray, axis: int, keepdims: bool, out: SparseArray):
    reduced_shape = tuple([s for i, s in enumerate(a.shape) if i != axis])
    reduced_strides = compact_strides(reduced_shape)
    reduced_indices = np.delete(a.indices, axis, axis=1)
    order = np.argsort((reduced_indices * reduced_strides).sum(axis=1))
    unique, split_inds, counts = np.unique(reduced_indices[order], axis=0,
                                           return_index=True, return_counts=True)
    if keepdims:
        out.indices = np.insert(unique, axis, 0, axis=1)
    else:
        out.indices = unique

    splits = np.split(a.values[order], split_inds[1:])
    max_len = np.max(counts)
    splits = [np.pad(s, (0, max_len - counts[i]), constant_values=float("-inf")) for i, s in enumerate(splits)]
    out.values = np.maximum(np.max(splits, axis=1), a.default_val)

    out.size = out.values.size
    out.default_val = a.default_val

