# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import numpy as np
import itertools

from numba_mlir.decorators import njit


def gen_params(args):
    params = {"empty": ()}
    params = {"1": (args[0],)}
    for i in [2, len(args) // 2, len(args)]:
        params[f"{i}-same"] = (args[0],) * i
        params[f"{i}-different"] = args[:i]

    return params


_trivial_args = (
    None,
    True,
    False,
    1,
    2.3,
    4.5 + 6.7j,
    (),
    (True),
    (False, 1, 2.5),
    ((None, True), (1, 2.3)),
    (
        (None, False, True),
        (
            1,
            2.3,
            4.5 + 6.7j,
        ),
    ),
    (
        (None, False),
        (True, 1),
        (
            2.3,
            4.5 + 6.7j,
        ),
    ),
)

_trivial_params = gen_params(_trivial_args)


class DispatcherTrivialArgs:
    version = "base"

    params = list(_trivial_params.keys())
    param_names = ["preset"]

    def setup(self, preset):
        def func(*args):
            pass

        self.func = njit(func)
        self.args = _trivial_params[preset]
        self.func(*self.args)

    def time_dispatcher(self, preset):
        self.func(*self.args)


def _get_dummy_func(i):
    def dummy():
        return i

    return dummy


_func_params = gen_params([njit(_get_dummy_func(i)) for i in range(12)])


class DispatcherFuncArg:
    version = "base"

    params = list(_func_params.keys())
    param_names = ["preset"]

    def setup(self, preset):
        def func(*args):
            pass

        self.func = njit(func)
        self.args = _func_params[preset]
        self.func(*self.args)

    def time_dispatcher(self, preset):
        self.func(*self.args)


_shapes = [(0,), (1,), (5,), (2, 3), (2, 3, 4), (1, 2, 1, 3, 1, 4, 1)]
_dtypes = [np.int64, np.float32]
_array_args = list(
    map(lambda t: np.empty(t[0], dtype=t[1]), itertools.product(_shapes, _dtypes))
)
_array_params = gen_params([njit(_get_dummy_func(i)) for i in range(12)])


class DispatcherArrayArg:
    version = "base"

    params = list(_array_params.keys())
    param_names = ["preset"]

    def setup(self, preset):
        def func(*args):
            pass

        self.func = njit(func)
        self.args = _array_params[preset]
        self.func(*self.args)

    def time_dispatcher(self, preset):
        self.func(*self.args)
