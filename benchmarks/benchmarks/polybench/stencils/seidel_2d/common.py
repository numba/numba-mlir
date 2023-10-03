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


def initialize(TSTEPS, N, datatype=np.float64):
    A = np.fromfunction(lambda i, j: (i * (j + 2) + 2) / N, (N, N), dtype=datatype)

    return TSTEPS, N, A


def get_impl(ctx):
    jit = ctx.jit
    np = ctx.numpy
    prange = ctx.prange

    @jit
    def kernel(TSTEPS, N, A):
        for t in range(0, TSTEPS - 1):
            for i in range(1, N - 1):
                A[i, 1:-1] += (
                    A[i - 1, :-2]
                    + A[i - 1, 1:-1]
                    + A[i - 1, 2:]
                    + A[i, 2:]
                    + A[i + 1, :-2]
                    + A[i + 1, 1:-1]
                    + A[i + 1, 2:]
                )
                for j in range(1, N - 1):
                    A[i, j] += A[i, j - 1]
                    A[i, j] /= 9.0

    def wrapper(TSTEPS, N, A):
        kernel(TSTEPS, N, A)
        return A

    return wrapper
