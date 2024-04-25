import numpy as np
import pytest
import random
import needle as ndl
from needle import backend_sparse as nd


_DEVICES = [
    nd.sparse_py(),
    nd.sparse_cpu(),
    pytest.param(
        nd.sparse_cuda(), marks=pytest.mark.skipif(not nd.sparse_cuda().enabled(), reason="No GPU")
    ),
]


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu_py", "cpu", "gpu"])
@pytest.mark.parametrize("shape", [(4, 5, 6), (2, 3, 4, 5), (1, 1, 1), (3, 4, 5, 6, 7)])
def test_identity(shape, device):
    _A = np.random.randn(*shape)
    A = nd.array(_A, device=device)
    np.testing.assert_allclose(_A, A.numpy(), atol=1e-5, rtol=1e-5)


# TODO test permute, broadcast_to, reshape, getitem, some combinations thereof
@pytest.mark.parametrize(
    "params",
    [
        {
            "shape": (4, 4),
            "np_fn": lambda X: X.transpose(),
            "nd_fn": lambda X: X.permute((1, 0)),
        },
        {
            "shape": (4, 1, 4),
            "np_fn": lambda X: np.broadcast_to(X, shape=(4, 5, 4)),
            "nd_fn": lambda X: X.broadcast_to((4, 5, 4)),
        },
        {
            "shape": (4, 3),
            "np_fn": lambda X: X.reshape(2, 2, 3),
            "nd_fn": lambda X: X.reshape((2, 2, 3)),
        },
        {
            "shape": (16, 16),  # testing for compaction of large ndims array
            "np_fn": lambda X: X.reshape(2, 4, 2, 2, 2, 2, 2),
            "nd_fn": lambda X: X.reshape((2, 4, 2, 2, 2, 2, 2)),
        },
        {
            "shape": (
                2,
                4,
                2,
                2,
                2,
                2,
                2,
            ),  # testing for compaction of large ndims array
            "np_fn": lambda X: X.reshape(16, 16),
            "nd_fn": lambda X: X.reshape((16, 16)),
        },
        {"shape": (8, 8), "np_fn": lambda X: X[4:, 4:], "nd_fn": lambda X: X[4:, 4:]},
        {
            "shape": (8, 8, 2, 2, 2, 2),
            "np_fn": lambda X: X[1:3, 5:8, 1:2, 0:1, 0:1, 1:2],
            "nd_fn": lambda X: X[1:3, 5:8, 1:2, 0:1, 0:1, 1:2],
        },
        {
            "shape": (7, 8),
            "np_fn": lambda X: X.transpose()[3:7, 2:5],
            "nd_fn": lambda X: X.permute((1, 0))[3:7, 2:5],
        },
    ],
    ids=[
        "transpose",
        "broadcast_to",
        "reshape1",
        "reshape2",
        "reshape3",
        "getitem1",
        "getitem2",
        "transposegetitem",
    ],
)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu_py", "cpu", "gpu"])
def test_compact(params, device):
    shape, np_fn, nd_fn = params["shape"], params["np_fn"], params["nd_fn"]
    _A = np.random.randint(low=0, high=10, size=shape)
    A = nd.array(_A, device=device)

    lhs = nd_fn(A).compact()
    assert lhs.is_compact(), "array is not compact"

    rhs = np_fn(_A)
    np.testing.assert_allclose(lhs.numpy(), rhs, atol=1e-5, rtol=1e-5)


reduce_params = [
    {"dims": (10,), "axis": 0},
    {"dims": (3, 5), "axis": 0},
    {"dims": (3, 5), "axis": 1},
    {"dims": (4, 5, 6), "axis": 0},
    {"dims": (4, 5, 6), "axis": 1},
    {"dims": (4, 5, 6), "axis": 2},
]


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu_py", "cpu", "gpu"])
@pytest.mark.parametrize("params", reduce_params)
@pytest.mark.parametrize("keepdims", [True, False])
def test_reduce_sum(params, device, keepdims):
    dims, axis = params["dims"], params["axis"]
    _A = np.random.randn(*dims)
    A = nd.array(_A, device=device)
    np.testing.assert_allclose(
        _A.sum(axis=axis, keepdims=keepdims), A.sum(axis=axis, keepdims=keepdims).numpy(), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu_py", "cpu", "gpu"])
@pytest.mark.parametrize("params", reduce_params)
@pytest.mark.parametrize("keepdims", [True, False])
def test_reduce_max(params, device, keepdims):
    dims, axis = params["dims"], params["axis"]
    _A = np.random.randn(*dims)
    A = nd.array(_A, device=device)
    np.testing.assert_allclose(
        _A.max(axis=axis, keepdims=keepdims), A.max(axis=axis, keepdims=keepdims).numpy(), atol=1e-5, rtol=1e-5
    )


""" For converting slice notation to slice objects to make some proceeding tests easier to read """


class _ShapeAndSlices(nd.NDArray):
    def __getitem__(self, idxs):
        idxs = tuple(
            [
                self.process_slice(s, i) if isinstance(s, slice) else slice(s, s + 1, 1)
                for i, s in enumerate(idxs)
            ]
        )
        return self.shape, idxs


ShapeAndSlices = lambda *shape: _ShapeAndSlices(np.ones(shape))


@pytest.mark.parametrize(
    "params",
    [
        {
            "lhs": ShapeAndSlices(4, 5, 6)[1:2, 0, 0],
            "rhs": ShapeAndSlices(7, 7, 7)[1:2, 0, 0],
        },
        {
            "lhs": ShapeAndSlices(4, 5, 6)[1:4:2, 0, 0],
            "rhs": ShapeAndSlices(7, 7, 7)[1:3, 0, 0],
        },
        {
            "lhs": ShapeAndSlices(4, 5, 6)[1:3, 2:5, 2:6],
            "rhs": ShapeAndSlices(7, 7, 7)[:2, :3, :4],
        },
        {
            "lhs": ShapeAndSlices(4, 5, 6)[1:3, 2:5, 2:6],
            "rhs": ShapeAndSlices(7, 7, 7)[1:3, 2:5, 2:6],
        },
    ],
)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu_py", "cpu", "gpu"])
def test_setitem_ewise(params, device):
    lhs_shape, lhs_slices = params["lhs"]
    rhs_shape, rhs_slices = params["rhs"]
    _A = np.random.randn(*lhs_shape)
    _B = np.random.randn(*rhs_shape)
    A = nd.array(_A, device=device)
    B = nd.array(_B, device=device)
    A[lhs_slices] = B[rhs_slices]
    _A[lhs_slices] = _B[rhs_slices]
    C = A.numpy()
    np.testing.assert_allclose(A.numpy(), _A, atol=1e-5, rtol=1e-5)


# Ex: We want arrays of size (4, 5, 6) setting element(s) [1:4, 2, 3] to a scalar
@pytest.mark.parametrize(
    "params",
    [
        ShapeAndSlices(4, 5, 6)[1, 2, 3],
        ShapeAndSlices(4, 5, 6)[1:4, 2, 3],
        ShapeAndSlices(4, 5, 6)[:4, 2:5, 3],
        ShapeAndSlices(4, 5, 6)[1::2, 2:5, ::2],
    ],
)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu_py", "cpu", "gpu"])
def test_setitem_scalar(params, device):
    shape, slices = params
    _A = np.random.randn(*shape)
    A = nd.array(_A, device=device)
    # probably tear these out using lambdas
    print(slices)
    _A[slices] = 4.0
    A[slices] = 4.0
    np.testing.assert_allclose(A.numpy(), _A, atol=1e-5, rtol=1e-5)



OPS = {
    "multiply": lambda a, b: a * b,
    "divide": lambda a, b: a / b,
    "add": lambda a, b: a + b,
    "radd": lambda a, b: b + a,
    "subtract": lambda a, b: a - b,
    "rsubtract": lambda a, b: b - a,
    "equal": lambda a, b: a == b,
    "not_equal": lambda a, b: a != b,
    "greater_than": lambda a, b: a > b,
    "less_than": lambda a, b: a < b,
    "greater_than_equal": lambda a, b: a >= b,
    "less_than_equal": lambda a, b: a <= b,
}
OP_FNS = [OPS[k] for k in OPS]
OP_NAMES = [k for k in OPS]

ewise_shapes = [(1, 1, 1), (4, 5, 6)]

@pytest.mark.parametrize("fn", OP_FNS, ids=OP_NAMES)
@pytest.mark.parametrize("shape", ewise_shapes)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu_py", "cpu", "gpu"])
def test_scalar_fn(fn, shape, device):
    _A = np.random.randn(*shape)
    A = nd.array(_A, device=device)
    s = random.uniform(-10000, 10000)
    np.testing.assert_allclose(fn(_A, s), fn(A, s).numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("fn", OP_FNS, ids=OP_NAMES)
@pytest.mark.parametrize("shape", ewise_shapes)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu_py", "cpu", "gpu"])
def test_ewise_fn(fn, shape, device):
    _A = np.random.randn(*shape)
    _B = np.random.randn(*shape)
    A = nd.array(_A, device=device)
    B = nd.array(_B, device=device)
    np.testing.assert_allclose(fn(_A, _B), fn(A, B).numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("shape", ewise_shapes)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu_py", "cpu", "gpu"])
def test_ewise_max(shape, device):
    _A = np.random.randn(*shape)
    _B = np.random.randn(*shape)
    A = nd.array(_A, device=device)
    B = nd.array(_B, device=device)
    np.testing.assert_allclose(
        np.maximum(_A, _B), A.maximum(B).numpy(), atol=1e-5, rtol=1e-5
    )


permute_params = [
    {"dims": (4, 5, 6), "axes": (0, 1, 2)},
    {"dims": (4, 5, 6), "axes": (1, 0, 2)},
    {"dims": (4, 5, 6), "axes": (2, 1, 0)},
]


@pytest.mark.parametrize("params", permute_params)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu_py", "cpu", "gpu"])
def test_permute(device, params):
    dims = params["dims"]
    axes = params["axes"]
    _A = np.random.randn(*dims)
    A = nd.array(_A, device=device)
    lhs = np.transpose(_A, axes=axes)
    rhs = A.permute(axes)
    np.testing.assert_allclose(lhs, rhs.numpy(), atol=1e-5, rtol=1e-5)


reshape_params = [
    {"shape": (8, 16), "new_shape": (2, 4, 16)},
    {"shape": (8, 16), "new_shape": (8, 4, 2, 2)},
]


@pytest.mark.parametrize("params", reshape_params)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu_py", "cpu", "gpu"])
def test_reshape(device, params):
    shape = params["shape"]
    new_shape = params["new_shape"]
    _A = np.random.randn(*shape)
    A = nd.array(_A, device=device)
    lhs = _A.reshape(*new_shape)
    rhs = A.reshape(new_shape)
    np.testing.assert_allclose(lhs, rhs.numpy(), atol=1e-5, rtol=1e-5)


getitem_params = [
    {"shape": (8, 16), "fn": lambda X: X[3:4, 3:4]},
    {"shape": (8, 16), "fn": lambda X: X[1:2, 1:3]},
    {"shape": (8, 16), "fn": lambda X: X[3:4, 1:4]},
    {"shape": (8, 16), "fn": lambda X: X[1:4, 3:4]},
]


@pytest.mark.parametrize("params", getitem_params)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu_py", "cpu", "gpu"])
def test_getitem(device, params):
    shape = params["shape"]
    fn = params["fn"]
    _A = np.random.randn(5, 5)
    A = nd.array(_A, device=device)
    lhs = fn(_A)
    rhs = fn(A)
    np.testing.assert_allclose(lhs, rhs.numpy(), atol=1e-5, rtol=1e-5)


broadcast_params = [
    {"from_shape": (1, 3, 4), "to_shape": (6, 3, 4)},
]


@pytest.mark.parametrize("params", broadcast_params)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu_py", "cpu", "gpu"])
def test_broadcast_to(device, params):
    from_shape, to_shape = params["from_shape"], params["to_shape"]
    _A = np.random.randn(*from_shape)
    A = nd.array(_A, device=device)
    lhs = np.broadcast_to(_A, shape=to_shape)
    rhs = A.broadcast_to(to_shape)
    np.testing.assert_allclose(lhs, rhs.numpy(), atol=1e-5, rtol=1e-5)


matmul_dims = [
    (8, 8, 8),
    (1, 2, 3),
    (3, 4, 5),
    (5, 4, 3),
    (16, 16, 16),
    (64, 64, 64),
    (72, 72, 72),
    (72, 73, 74),
    (74, 73, 72),
    (128, 128, 128),
]


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu_py", "cpu", "gpu"])
@pytest.mark.parametrize("m,n,p", matmul_dims)
def test_matmul(m, n, p, device):
    _A = np.random.randn(m, n)
    _B = np.random.randn(n, p)
    A = nd.array(_A, device=device)
    B = nd.array(_B, device=device)
    np.testing.assert_allclose((A @ B).numpy(), _A @ _B, atol=1e-3)


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu_py", "cpu", "gpu"])
def test_scalar_mul(device):
    A = np.random.randn(5, 5)
    B = nd.array(A, device=device)
    np.testing.assert_allclose(A * 5.0, (B * 5.0).numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu_py", "cpu", "gpu"])
def test_scalar_div(device):
    A = np.random.randn(5, 5)
    B = nd.array(A, device=device)
    np.testing.assert_allclose(A / 5.0, (B / 5.0).numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu_py", "cpu", "gpu"])
def test_scalar_power(device):
    A = np.random.randn(5, 5)
    B = nd.array(A, device=device)
    np.testing.assert_allclose(
        np.power(A, 5.0), (B**5.0).numpy(), atol=1e-5, rtol=1e-5
    )
    np.testing.assert_allclose(
        np.power(A, 0.5), (B**0.5).numpy(), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu_py", "cpu", "gpu"])
def test_scalar_maximum(device):
    A = np.random.randn(5, 5)
    B = nd.array(A, device=device)
    C = (np.max(A) + 1.0).item()
    np.testing.assert_allclose(
        np.maximum(A, C), (B.maximum(C)).numpy(), atol=1e-5, rtol=1e-5
    )
    C = (np.max(A) - 1.0).item()
    np.testing.assert_allclose(
        np.maximum(A, C), (B.maximum(C)).numpy(), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu_py", "cpu", "gpu"])
def test_scalar_eq(device):
    A = np.random.randn(5, 5)
    B = nd.array(A, device=device)
    C = A[0, 1].item()
    np.testing.assert_allclose(A == C, (B == C).numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu_py", "cpu", "gpu"])
def test_scalar_ge(device):
    A = np.random.randn(5, 5)
    B = nd.array(A, device=device)
    C = A[0, 1].item()
    np.testing.assert_allclose(A >= C, (B >= C).numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu_py", "cpu", "gpu"])
def test_ewise_neg(device):
    A = np.abs(np.random.randn(5, 5))
    B = nd.array(A, device=device)
    np.testing.assert_allclose(-A, (-B).numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu_py", "cpu", "gpu"])
def test_ewise_log(device):
    A = np.abs(np.random.randn(5, 5))
    B = nd.array(A, device=device)
    np.testing.assert_allclose(np.log(A), (B.log()).numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu_py", "cpu", "gpu"])
def test_ewise_exp(device):
    A = np.random.randn(5, 5)
    B = nd.array(A, device=device)
    np.testing.assert_allclose(np.exp(A), (B.exp()).numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu_py", "cpu", "gpu"])
def test_ewise_tanh(device):
    A = np.random.randn(5, 5)
    B = nd.array(A, device=device)
    np.testing.assert_allclose(np.tanh(A), (B.tanh()).numpy(), atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    print("You have to run the tests with pytest due to parameterization.")