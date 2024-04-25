"""
Throughout our code we closely follow the conventions of our existing Needle
project software architecture. This is the main entry point for the sparse
backend and it is responsible for importing the sparse backend and exposing
the sparse API to the rest of the codebase.

The sparse_ndarray.py file contains the main endpoints for the sparse backend.
Just like the dense backend, the sparse backend exposes a single class
SparseNDArray that is responsible for all the sparse operations.
"""

from .sparse_ndarray import *