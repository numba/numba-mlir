# SPDX-FileCopyrightText: 2010-2016 Ohio State University
# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause


from numba_mlir.mlir.benchmarking import parse_config, filter_presets
from os import path
import numpy as np

config = parse_config(path.join(path.dirname(__file__), "config.toml"))
parameters = dict(config["benchmark"]["parameters"])
presets = filter_presets(parameters.keys())


def initialize(N, datatype=np.float64):
    L = np.fromfunction(lambda i, j: (i + N - j + 1) * 2 / N, (N, N), dtype=datatype)
    x = np.full((N,), -999, dtype=datatype)
    b = np.fromfunction(lambda i: i, (N,), dtype=datatype)

    return L, x, b


def get_impl(ctx):
    jit = ctx.jit
    np = ctx.numpy
    prange = ctx.prange

    @jit
    def kernel(L, x, b):
        for i in range(x.shape[0]):
            x[i] = (b[i] - L[i, :i] @ x[:i]) / L[i, i]

    def wrapper(L, x, b):
        kernel(L, x, b)
        return x

    return wrapper
