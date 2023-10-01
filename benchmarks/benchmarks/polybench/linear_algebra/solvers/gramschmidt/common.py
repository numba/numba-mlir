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
    from numpy.random import default_rng

    rng = default_rng(42)

    A = rng.random((M, N), dtype=datatype)
    while np.linalg.matrix_rank(A) < N:
        A = rng.random((M, N), dtype=datatype)

    return (A,)


def get_impl(ctx):
    jit = ctx.jit
    np = ctx.numpy
    prange = ctx.prange

    def kernel(A):
        Q = np.zeros_like(A)
        R = np.zeros((A.shape[1], A.shape[1]), dtype=A.dtype)

        for k in range(A.shape[1]):
            nrm = np.dot(A[:, k], A[:, k])
            R[k, k] = np.sqrt(nrm)
            Q[:, k] = A[:, k] / R[k, k]
            for j in range(k + 1, A.shape[1]):
                R[k, j] = np.dot(Q[:, k], A[:, j])
                A[:, j] -= Q[:, k] * R[k, j]

        return Q, R

    return jit(kernel)
