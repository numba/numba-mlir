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


def initialize(N, datatype=np.int32):
    path = np.fromfunction(lambda i, j: i * j % 7 + 1, (N, N), dtype=datatype)
    for i in range(N):
        for j in range(N):
            if (i + j) % 13 == 0 or (i + j) % 7 == 0 or (i + j) % 11 == 0:
                path[i, j] = 999

    return (path,)


def get_impl(ctx):
    jit = ctx.jit
    np = ctx.numpy
    prange = ctx.prange

    @jit
    def kernel(path):
        for k in range(path.shape[0]):
            path[:] = np.minimum(path[:], np.add.outer(path[:, k], path[k, :]))

    def wrapper(path):
        kernel(path)
        return path

    return wrapper


def get_impl_numba(ctx):
    jit = ctx.jit
    np = ctx.numpy
    prange = ctx.prange

    @jit
    def kernel(path):
        for k in range(path.shape[0]):
            # path[:] = np.minimum(path[:], np.add.outer(path[:, k], path[k, :]))
            for i in prange(path.shape[0]):
                path[i, :] = np.minimum(path[i, :], path[i, k] + path[k, :])

    def wrapper(path):
        kernel(path)
        return path

    return wrapper
