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


def initialize(NI, NJ, NK, datatype=np.float64):
    alpha = datatype(1.5)
    beta = datatype(1.2)
    C = np.fromfunction(lambda i, j: ((i * j + 1) % NI) / NI, (NI, NJ), dtype=datatype)
    A = np.fromfunction(lambda i, k: (i * (k + 1) % NK) / NK, (NI, NK), dtype=datatype)
    B = np.fromfunction(lambda k, j: (k * (j + 2) % NJ) / NJ, (NK, NJ), dtype=datatype)

    return alpha, beta, C, A, B


def get_impl(ctx):
    jit = ctx.jit
    np = ctx.numpy
    prange = ctx.prange

    @jit
    def kernel(alpha, beta, C, A, B):
        C[:] = alpha * A @ B + beta * C

    def wrapper(alpha, beta, C, A, B):
        kernel(alpha, beta, C, A, B)
        return C

    return wrapper
