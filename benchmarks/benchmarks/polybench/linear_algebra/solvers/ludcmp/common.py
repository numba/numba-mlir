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
    A = np.empty((N, N), dtype=datatype)
    for i in range(N):
        A[i, : i + 1] = np.fromfunction(
            lambda j: (-j % N) / N + 1, (i + 1,), dtype=datatype
        )
        A[i, i + 1 :] = 0.0
        A[i, i] = 1.0
    A[:] = A @ np.transpose(A)
    fn = datatype(N)
    b = np.fromfunction(lambda i: (i + 1) / fn / 2.0 + 4.0, (N,), dtype=datatype)

    return A, b


def get_impl(ctx):
    jit = ctx.jit
    np = ctx.numpy
    prange = ctx.prange

    def kernel(A, b):
        x = np.zeros_like(b)
        y = np.zeros_like(b)

        for i in range(A.shape[0]):
            for j in range(i):
                A[i, j] -= A[i, :j] @ A[:j, j]
                A[i, j] /= A[j, j]
            for j in range(i, A.shape[0]):
                A[i, j] -= A[i, :i] @ A[:i, j]
        for i in range(A.shape[0]):
            y[i] = b[i] - A[i, :i] @ y[:i]
        for i in range(A.shape[0] - 1, -1, -1):
            x[i] = (y[i] - A[i, i + 1 :] @ x[i + 1 :]) / A[i, i]

        return x, y

    return jit(kernel)
