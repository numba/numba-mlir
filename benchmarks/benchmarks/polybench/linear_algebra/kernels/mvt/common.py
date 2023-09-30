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
    x1 = np.fromfunction(lambda i: (i % N) / N, (N,), dtype=datatype)
    x2 = np.fromfunction(lambda i: ((i + 1) % N) / N, (N,), dtype=datatype)
    y_1 = np.fromfunction(lambda i: ((i + 3) % N) / N, (N,), dtype=datatype)
    y_2 = np.fromfunction(lambda i: ((i + 4) % N) / N, (N,), dtype=datatype)
    A = np.fromfunction(lambda i, j: (i * j % N) / N, (N, N), dtype=datatype)

    return x1, x2, y_1, y_2, A


def get_impl(ctx):
    jit = ctx.jit
    np = ctx.numpy
    prange = ctx.prange

    @jit
    def kernel(x1, x2, y_1, y_2, A):
        x1 += A @ y_1
        x2 += y_2 @ A

    def wrapper(x1, x2, y_1, y_2, A):
        kernel(x1, x2, y_1, y_2, A)
        return x1, x2

    return wrapper
