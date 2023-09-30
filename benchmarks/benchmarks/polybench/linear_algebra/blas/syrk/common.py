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


def initialize(M, N, datatype=np.float64):
    alpha = datatype(1.5)
    beta = datatype(1.2)
    C = np.fromfunction(lambda i, j: ((i * j + 2) % N) / M, (N, N), dtype=datatype)
    A = np.fromfunction(lambda i, j: ((i * j + 1) % N) / N, (N, M), dtype=datatype)

    return alpha, beta, C, A


def get_impl(ctx):
    jit = ctx.jit
    np = ctx.numpy
    prange = ctx.prange

    @jit
    def kernel(alpha, beta, C, A):
        for i in range(A.shape[0]):
            C[i, : i + 1] *= beta
            for k in range(A.shape[1]):
                C[i, : i + 1] += alpha * A[i, k] * A[: i + 1, k]

    def wrapper(alpha, beta, C, A):
        kernel(alpha, beta, C, A)
        return C

    return wrapper
