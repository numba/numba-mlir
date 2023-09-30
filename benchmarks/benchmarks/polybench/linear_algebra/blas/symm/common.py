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
    C = np.fromfunction(lambda i, j: ((i + j) % 100) / M, (M, N), dtype=datatype)
    B = np.fromfunction(lambda i, j: ((N + i - j) % 100) / M, (M, N), dtype=datatype)
    A = np.empty((M, M), dtype=datatype)
    for i in range(M):
        A[i, : i + 1] = np.fromfunction(
            lambda j: ((i + j) % 100) / M, (i + 1,), dtype=datatype
        )
        A[i, i + 1 :] = -999

    return alpha, beta, C, A, B


def get_impl(ctx):
    jit = ctx.jit
    np = ctx.numpy
    prange = ctx.prange

    @jit
    def kernel(alpha, beta, C, A, B):
        temp2 = np.empty((C.shape[1],), dtype=C.dtype)
        C *= beta
        for i in range(C.shape[0]):
            for j in prange(C.shape[1]):
                C[:i, j] += alpha * B[i, j] * A[i, :i]
                temp2[j] = B[:i, j] @ A[i, :i]
            C[i, :] += alpha * B[i, :] * A[i, i] + alpha * temp2

    def wrapper(alpha, beta, C, A, B):
        kernel(alpha, beta, C, A, B)
        return C

    return wrapper
