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
    r = np.fromfunction(lambda i: N + 1 - i, (N,), dtype=datatype)
    return (r,)


def get_impl(ctx):
    jit = ctx.jit
    np = ctx.numpy
    prange = ctx.prange

    def kernel(r):
        y = np.empty_like(r)
        alpha = -r[0]
        beta = 1.0
        y[0] = -r[0]

        for k in range(1, r.shape[0]):
            beta *= 1.0 - alpha * alpha
            alpha = -(r[k] + np.dot(np.flip(r[:k]), y[:k])) / beta
            y[:k] += alpha * np.flip(y[:k])
            y[k] = alpha

        return y

    return jit(kernel)
