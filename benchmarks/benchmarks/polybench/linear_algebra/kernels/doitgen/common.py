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


def initialize(NR, NQ, NP, datatype=np.float64):
    A = np.fromfunction(
        lambda i, j, k: ((i * j + k) % NP) / NP, (NR, NQ, NP), dtype=datatype
    )
    C4 = np.fromfunction(lambda i, j: (i * j % NP) / NP, (NP, NP), dtype=datatype)

    return NR, NQ, NP, A, C4


def get_impl(ctx):
    jit = ctx.jit
    np = ctx.numpy
    prange = ctx.prange

    @jit
    def kernel(NR, NQ, NP, A, C4):
        A[:] = np.reshape(np.reshape(A, (NR, NQ, 1, NP)) @ C4, (NR, NQ, NP))

    def wrapper(NR, NQ, NP, A, C4):
        kernel(NR, NQ, NP, A, C4)
        return A

    return wrapper


def get_impl_numba(ctx):
    jit = ctx.jit
    np = ctx.numpy
    prange = ctx.prange

    @jit
    def kernel(NR, NQ, NP, A, C4):
        for r in range(NR):
            for q in range(NQ):
                tmp = A[r, q, :] @ C4
                A[r, q, :] = tmp

    def wrapper(NR, NQ, NP, A, C4):
        kernel(NR, NQ, NP, A, C4)
        return A

    return wrapper
