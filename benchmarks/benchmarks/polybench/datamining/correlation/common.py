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
    float_n = datatype(N)
    data = np.fromfunction(lambda i, j: (i * j) / M + i, (N, M), dtype=datatype)

    return M, float_n, data


def get_impl(ctx):
    jit = ctx.jit
    np = ctx.numpy
    prange = ctx.prange

    def kernel(M, float_n, data):
        mean = np.mean(data, axis=0)
        stddev = np.std(data, axis=0)
        stddev[stddev <= 0.1] = 1.0
        data -= mean
        data /= np.sqrt(float_n) * stddev
        corr = np.eye(M, dtype=data.dtype)
        for i in prange(M - 1):
            corr[i + 1 : M, i] = corr[i, i + 1 : M] = data[:, i] @ data[:, i + 1 : M]

        return corr

    return jit(kernel)


def get_impl_numba(ctx):
    jit = ctx.jit
    np = ctx.numpy
    prange = ctx.prange

    def kernel(M, float_n, data):
        # mean = np.mean(data, axis=0)
        mean = np.sum(data, axis=0) / float_n
        # stddev = np.std(data, axis=0)
        stddev = np.sqrt(np.sum((data - mean) ** 2, axis=0) / float_n)
        stddev[stddev <= 0.1] = 1.0
        data -= mean
        data /= np.sqrt(float_n) * stddev
        corr = np.eye(M, dtype=data.dtype)
        for i in prange(M - 1):
            corr[i + 1 : M, i] = corr[i, i + 1 : M] = data[:, i] @ data[:, i + 1 : M]

        return corr

    return jit(kernel)
