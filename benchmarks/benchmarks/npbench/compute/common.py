# SPDX-FileCopyrightText: 2023 Stefan Behnel, Robert Bradshaw,
#   Dag Sverre Seljebotn, Greg Ewing, William Stein, Gabriel Gellner, et al.
# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

# https://cython.readthedocs.io/en/latest/src/userguide/numpy_tutorial.html

from numba_mlir.mlir.benchmarking import parse_config, filter_presets
from os import path

config = parse_config(path.join(path.dirname(__file__), "config.toml"))
parameters = dict(config["benchmark"]["parameters"])
presets = filter_presets(parameters.keys())


def initialize(M, N):
    import numpy as np
    from numpy.random import default_rng

    rng = default_rng(42)
    array_1 = rng.uniform(0, 1000, size=(M, N)).astype(np.int64)
    array_2 = rng.uniform(0, 1000, size=(M, N)).astype(np.int64)
    a = np.int64(4)
    b = np.int64(3)
    c = np.int64(9)
    return array_1, array_2, a, b, c


def get_impl(ctx):
    jit = ctx.jit
    np = ctx.numpy
    prange = ctx.prange

    def compute(array_1, array_2, a, b, c):
        return np.clip(array_1, 2, 10) * a + array_2 * b + c

    return jit(compute)
