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
    seq = np.fromfunction(lambda i: (i + 1) % 4, (N,), dtype=datatype)

    return N, seq


def get_impl(ctx):
    jit = ctx.jit
    np = ctx.numpy
    prange = ctx.prange

    @jit
    def match(b1, b2):
        if b1 + b2 == 3:
            return 1
        else:
            return 0

    def kernel(N, seq):
        table = np.zeros((N, N), np.int32)

        for i in range(N - 1, -1, -1):
            for j in range(i + 1, N):
                if j - 1 >= 0:
                    table[i, j] = max(table[i, j], table[i, j - 1])
                if i + 1 < N:
                    table[i, j] = max(table[i, j], table[i + 1, j])
                if j - 1 >= 0 and i + 1 < N:
                    if i < j - 1:
                        table[i, j] = max(
                            table[i, j], table[i + 1, j - 1] + match(seq[i], seq[j])
                        )
                    else:
                        table[i, j] = max(table[i, j], table[i + 1, j - 1])
                for k in range(i + 1, j):
                    table[i, j] = max(table[i, j], table[i, k] + table[k + 1, j])

        return table

    return jit(kernel)
