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
    alpha = datatype(1.5)
    beta = datatype(1.2)
    fn = datatype(N)
    A = np.fromfunction(lambda i, j: (i * j % N) / N, (N, N), dtype=datatype)
    u1 = np.fromfunction(lambda i: i, (N,), dtype=datatype)
    u2 = np.fromfunction(lambda i: ((i + 1) / fn) / 2.0, (N,), dtype=datatype)
    v1 = np.fromfunction(lambda i: ((i + 1) / fn) / 4.0, (N,), dtype=datatype)
    v2 = np.fromfunction(lambda i: ((i + 1) / fn) / 6.0, (N,), dtype=datatype)
    w = np.zeros((N,), dtype=datatype)
    x = np.zeros((N,), dtype=datatype)
    y = np.fromfunction(lambda i: ((i + 1) / fn) / 8.0, (N,), dtype=datatype)
    z = np.fromfunction(lambda i: ((i + 1) / fn) / 9.0, (N,), dtype=datatype)

    return alpha, beta, A, u1, v1, u2, v2, w, x, y, z


def get_impl(ctx):
    jit = ctx.jit
    np = ctx.numpy
    prange = ctx.prange

    @jit
    def kernel(alpha, beta, A, u1, v1, u2, v2, w, x, y, z):
        A += np.outer(u1, v1) + np.outer(u2, v2)
        x += beta * y @ A + z
        w += alpha * A @ x

    def wrapper(alpha, beta, A, u1, v1, u2, v2, w, x, y, z):
        kernel(alpha, beta, A, u1, v1, u2, v2, w, x, y, z)
        return A, w, x

    return wrapper
