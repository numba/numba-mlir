# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

# Sparse Matrix-Vector Multiplication (SpMV)

from numba_mlir.mlir.benchmarking import parse_config, filter_presets
from os import path

config = parse_config(path.join(path.dirname(__file__), "config.toml"))
parameters = dict(config["benchmark"]["parameters"])
presets = filter_presets(parameters.keys())


def initialize(M, N, nnz):
    import numpy as np
    from numpy.random import default_rng

    rng = default_rng(42)

    x = rng.random((N,))

    from scipy.sparse import random

    matrix = random(
        M,
        N,
        density=nnz / (M * N),
        format="csr",
        dtype=np.float64,
        random_state=rng,
    )
    rows = np.uint32(matrix.indptr)
    cols = np.uint32(matrix.indices)
    vals = matrix.data

    return rows, cols, vals, x


def get_impl(ctx):
    jit = ctx.jit
    np = ctx.numpy
    prange = ctx.prange

    # Matrix-Vector Multiplication with the matrix given in Compressed Sparse Row
    # (CSR) format
    def spmv(A_row, A_col, A_val, x):
        y = np.empty(A_row.size - 1, A_val.dtype)

        for i in prange(A_row.size - 1):
            cols = A_col[A_row[i] : A_row[i + 1]]
            vals = A_val[A_row[i] : A_row[i + 1]]
            y[i] = vals @ x[cols]

        return y

    return jit(spmv)
