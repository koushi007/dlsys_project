"""
Logic for backend selection.

This module provides the logic for selecting the backend for the needle array library.
The backend is determined by the value of the environment variable NEEDLE_BACKEND.
The available backends are "nd" (needle backend), "sparse" (sparse backend), and "np" (numpy backend).

The selected backend is imported and the appropriate classes and functions are assigned to the
corresponding variables. The selected backend is also printed to the console.

If an unknown backend is specified, a RuntimeError is raised.
"""
import os


BACKEND = os.environ.get("NEEDLE_BACKEND", "nd")


if BACKEND == "nd":
    print("Using needle backend")
    from . import backend_ndarray as array_api
    from .backend_ndarray import (
        all_devices,
        cuda,
        cpu,
        cpu_numpy,
        default_device,
        BackendDevice as Device,
    )
    NDArray = array_api.NDArray

elif BACKEND == "nd_sparse":
    """
    This option allows users to use both the original needle backend and the
    sparse backend simultaneously. We support interoperability between the two
    backends implicitly (in operations between dense and sparse arrays) and
    explicitly allow users to convert between NDArray and SparseNDArray.
    """
    print("Using needle backend with sparse support")
    from . import backend_ndarray as array_api
    from .backend_ndarray import (
        all_devices,
        cuda,
        cpu,
        cpu_numpy,
        default_device,
        BackendDevice as Device,
    )
    NDArray = array_api.NDArray

    from . import backend_sparse as sparse_array_api
    from .backend_sparse import (
        sparse_all_devices,
        sparse_cuda,
        sparse_cpu,
        sparse_default_device,
        SparseBackendDevice as SparseDevice,
    )
    SparseNDArray = sparse_array_api.SparseNDArray

elif BACKEND == "sparse":
    """
    By setting the environment variable NEEDLE_BACKEND=sparse, we allow users
    to use the sparse backend as a drop-in replacement for the needle backend
    This is important because it expands the capabilities of sparse arrays to
    all models built with needle.
        Note that the sparse backend fully supports backpropagation as well.
        See auto_grad.py for details on how gradients are sparsified to
        preserve sparsity and ensure computations are efficient.
    """
    print("Using sparse backend")
    from . import backend_sparse as array_api
    from .backend_sparse import (
        sparse_all_devices as all_devices,
        sparse_cuda as cuda,
        sparse_cpu as cpu,
        sparse_default_device as default_device,
        SparseBackendDevice as Device,
    )

    NDArray = array_api.SparseNDArray
elif BACKEND == "np":
    print("Using numpy backend")
    import numpy as array_api
    from .backend_numpy import all_devices, cpu, default_device, Device

    NDArray = array_api.ndarray
else:
    raise RuntimeError("Unknown needle array backend %s" % BACKEND)
