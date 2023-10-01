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


def initialize(NI, NJ, NK, NL, datatype=np.float64):
    alpha = datatype(1.5)
    beta = datatype(1.2)
    A = np.fromfunction(lambda i, j: ((i * j + 1) % NI) / NI, (NI, NK), dtype=datatype)
    B = np.fromfunction(lambda i, j: (i * (j + 1) % NJ) / NJ, (NK, NJ), dtype=datatype)
    C = np.fromfunction(
        lambda i, j: ((i * (j + 3) + 1) % NL) / NL, (NJ, NL), dtype=datatype
    )
    D = np.fromfunction(lambda i, j: (i * (j + 2) % NK) / NK, (NI, NL), dtype=datatype)

    return alpha, beta, A, B, C, D


def get_impl(ctx):
    jit = ctx.jit
    np = ctx.numpy
    prange = ctx.prange

    @jit
    def kernel(alpha, beta, A, B, C, D):
        D[:] = alpha * A @ B @ C + beta * D

    def wrapper(alpha, beta, A, B, C, D):
        kernel(alpha, beta, A, B, C, D)
        return D

    return wrapper
