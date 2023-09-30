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
    A = np.fromfunction(lambda i, j: ((i * j) % M) / M, (M, M), dtype=datatype)
    for i in range(M):
        A[i, i] = 1.0
    B = np.fromfunction(lambda i, j: ((N + i - j) % N) / N, (M, N), dtype=datatype)

    return alpha, A, B


def get_impl(ctx):
    jit = ctx.jit
    np = ctx.numpy
    prange = ctx.prange

    @jit
    def kernel(alpha, A, B):
        for i in range(B.shape[0]):
            for j in prange(B.shape[1]):
                B[i, j] += np.dot(A[i + 1 :, i], B[i + 1 :, j])
        B *= alpha

    def wrapper(alpha, A, B):
        kernel(alpha, A, B)
        return B

    return wrapper
