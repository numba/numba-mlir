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
import timeit
from ..decorators import njit, njit_replace_parfors
from .utils import readenv
from numpy.testing import assert_allclose
import inspect
from asv_runner.benchmarks.mark import SkipNotImplemented

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


_rp_njit = njit_replace_parfors(parallel=True, fastmath=True)


def get_numba_replace_parfor_context():
    return _BenchmarkContext(_rp_njit, np, nb.prange)


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


TEST_PRESETS = set(
    filter(None, readenv("NUMBA_MLIR_BENCH_PRESETS", str, "S").split(","))
)


def filter_presets(presets):
    return [x for x in presets if x in TEST_PRESETS]


def copy_args(arg):
    if isinstance(arg, tuple):
        return tuple(map(copy_args, arg))

    if isinstance(arg, list):
        return lsit(map(copy_args, arg))

    if hasattr(arg, "__array__"):
        return arg.copy()

    return arg


VALIDATE = readenv("NUMBA_MLIR_BENCH_VALIDATE", int, 1)


def has_dpctl():
    try:
        import dpctl
    except ImportError:
        return False

    return True


def get_dpctl_devices():
    try:
        import dpctl
    except ImportError:
        return []

    return list(map(lambda d: d.filter_string, dpctl.get_devices()))


class BenchmarkBase:
    timer = timeit.default_timer
    version = "base"

    def __init__(self):
        self.is_validate = VALIDATE
        self.is_expected_failure = False
        func = self.get_func()

        def setup(*args, **kwargs):
            self.args = self.initialize(*args, **kwargs)
            try:
                res = func(*copy_args(self.args))
                if self.is_validate:
                    self.validate(copy_args(self.args), res)
            except:
                if self.is_validate and self.is_expected_failure:
                    raise SkipNotImplemented("Expected failure")
                else:
                    raise
            else:
                if self.is_validate and self.is_expected_failure:
                    raise ValueError("Unexpected success")

        setup.pretty_source = inspect.getsource(self.initialize)
        self.setup = setup

        def teardown(*args, **kwargs):
            if hasattr(self, "args"):
                del self.args

        self.teardown = teardown

        def time_benchmark(*arg, **kwargs):
            func(*self.args)

        time_benchmark.pretty_source = inspect.getsource(func)
        self.time_benchmark = time_benchmark

    def get_func(self, *args, **kwargs):
        raise SkipNotImplemented("No function was provided")

    def setup(self, *args, **kwargs):
        # Dummy method, will be overriden
        pass

    def teardown(self, *args, **kwargs):
        # Dummy method, will be overriden
        pass

    def time_benchmark(self, *args, **kwargs):
        # Dummy method, will be overriden
        pass
