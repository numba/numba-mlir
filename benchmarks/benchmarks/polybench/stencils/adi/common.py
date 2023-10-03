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


def initialize(TSTEPS, N, datatype=np.float64):
    u = np.fromfunction(lambda i, j: (i + N - j) / N, (N, N), dtype=datatype)

    return TSTEPS, N, u


def get_impl(ctx):
    jit = ctx.jit
    np = ctx.numpy
    prange = ctx.prange

    @jit
    def kernel(TSTEPS, N, u):
        v = np.empty(u.shape, dtype=u.dtype)
        p = np.empty(u.shape, dtype=u.dtype)
        q = np.empty(u.shape, dtype=u.dtype)

        DX = 1.0 / N
        DY = 1.0 / N
        DT = 1.0 / TSTEPS
        B1 = 2.0
        B2 = 1.0
        mul1 = B1 * DT / (DX * DX)
        mul2 = B2 * DT / (DY * DY)

        a = -mul1 / 2.0
        b = 1.0 + mul2
        c = a
        d = -mul2 / 2.0
        e = 1.0 + mul2
        f = d

        for t in range(1, TSTEPS + 1):
            v[0, 1 : N - 1] = 1.0
            p[1 : N - 1, 0] = 0.0
            q[1 : N - 1, 0] = v[0, 1 : N - 1]
            for j in range(1, N - 1):
                p[1 : N - 1, j] = -c / (a * p[1 : N - 1, j - 1] + b)
                q[1 : N - 1, j] = (
                    -d * u[j, 0 : N - 2]
                    + (1.0 + 2.0 * d) * u[j, 1 : N - 1]
                    - f * u[j, 2:N]
                    - a * q[1 : N - 1, j - 1]
                ) / (a * p[1 : N - 1, j - 1] + b)
            v[N - 1, 1 : N - 1] = 1.0
            for j in range(N - 2, 0, -1):
                v[j, 1 : N - 1] = (
                    p[1 : N - 1, j] * v[j + 1, 1 : N - 1] + q[1 : N - 1, j]
                )

            u[1 : N - 1, 0] = 1.0
            p[1 : N - 1, 0] = 0.0
            q[1 : N - 1, 0] = u[1 : N - 1, 0]
            for j in range(1, N - 1):
                p[1 : N - 1, j] = -f / (d * p[1 : N - 1, j - 1] + e)
                q[1 : N - 1, j] = (
                    -a * v[0 : N - 2, j]
                    + (1.0 + 2.0 * a) * v[1 : N - 1, j]
                    - c * v[2:N, j]
                    - d * q[1 : N - 1, j - 1]
                ) / (d * p[1 : N - 1, j - 1] + e)
            u[1 : N - 1, N - 1] = 1.0
            for j in range(N - 2, 0, -1):
                u[1 : N - 1, j] = (
                    p[1 : N - 1, j] * u[1 : N - 1, j + 1] + q[1 : N - 1, j]
                )

    def wrapper(TSTEPS, N, u):
        kernel(TSTEPS, N, u)
        return u

    return wrapper
