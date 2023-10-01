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


def initialize(NI, NJ, NK, NL, NM, datatype=np.float64):
    A = np.fromfunction(
        lambda i, j: ((i * j + 1) % NI) / (5 * NI), (NI, NK), dtype=datatype
    )
    B = np.fromfunction(
        lambda i, j: ((i * (j + 1) + 2) % NJ) / (5 * NJ),
        (NK, NJ),
        dtype=datatype,
    )
    C = np.fromfunction(
        lambda i, j: (i * (j + 3) % NL) / (5 * NL), (NJ, NM), dtype=datatype
    )
    D = np.fromfunction(
        lambda i, j: ((i * (j + 2) + 2) % NK) / (5 * NK),
        (NM, NL),
        dtype=datatype,
    )

    return A, B, C, D


def get_impl(ctx):
    jit = ctx.jit
    np = ctx.numpy
    prange = ctx.prange

    def kernel(A, B, C, D):
        return A @ B @ C @ D

    return jit(kernel)
