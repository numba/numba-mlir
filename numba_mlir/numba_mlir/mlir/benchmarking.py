# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


"""
Benchmarking utilities
TODO: Move to separate package
"""


from collections import namedtuple
import numpy as np
import numba as nb
from ..decorators import njit
from numpy.testing import assert_allclose


_BenchmarkContext = namedtuple(
    "_BenchmarkContext",
    [
        "jit",
        "numpy",
        "prange",
    ],
)


def get_numpy_context():
    return _BenchmarkContext(lambda a: a, np, range)


_nb_njit = nb.njit(parallel=True, fastmath=True)


def get_numba_context():
    return _BenchmarkContext(_nb_njit, np, nb.prange)


_nm_njit = njit(parallel=True, fastmath=True)


def get_numba_mlir_context():
    return _BenchmarkContext(_nm_njit, np, nb.prange)


def parse_config(file_path):
    import tomli

    with open(file_path) as file:
        file_contents = file.read()

    return tomli.loads(file_contents)


_seen_dpctl = False


def to_device(value, device):
    if isinstance(value, tuple):
        return tuple(to_device(v, device) for v in value)

    if device and isinstance(value, np.ndarray):
        global _seen_dpctl
        import dpctl.tensor as dpt

        _seen_dpctl = True
        if ref_array.flags["C_CONTIGUOUS"]:
            order = "C"
        elif ref_array.flags["F_CONTIGUOUS"]:
            order = "F"
        else:
            order = "K"
        return dpt.asarray(
            obj=ref_array,
            dtype=ref_array.dtype,
            device=self.sycl_device,
            copy=None,
            usm_type=None,
            sycl_queue=None,
            order=order,
        )

    return value


def from_device(value):
    if isinstance(value, tuple):
        return tuple(from_device(v) for v in value)

    if _seen_dpctl:
        import dpctl.tensor as dpt

        if isinstance(value, dpt._usmarray.usm_ndarray):
            return dpt.asnumpy(value)

    return value


def assert_allclose_recursive(actual, desired, rtol=1e-07, atol=0):
    if isinstance(actual, tuple):
        assert isinstance(desired, tuple)
        for a, b in zip(actual, desired):
            assert_allclose_recursive(a, b, rtol, atol)

        return

    assert_allclose(actual, desired, rtol, atol)


class BenchmarkBase:
    def get_func(self, *args, **kwargs):
        raise NotImplementedError

    def setup(self, *args, **kwargs):
        self.func = self.get_func(*args, **kwargs)
        self.args = self.initialize(*args, **kwargs)
        res = self.func(*self.args)
        self.validate(self.args, res)

    def teardown(self, *args, **kwargs):
        if hasattr(self, "args"):
            del self.args

        if hasattr(self, "func"):
            del self.func

    def time_benchmark(self, *args, **kwargs):
        self.func(*self.args)
