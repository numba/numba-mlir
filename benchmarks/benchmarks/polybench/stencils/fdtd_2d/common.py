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


def initialize(TMAX, NX, NY, datatype=np.float64):
    ex = np.fromfunction(lambda i, j: (i * (j + 1)) / NX, (NX, NY), dtype=datatype)
    ey = np.fromfunction(lambda i, j: (i * (j + 2)) / NY, (NX, NY), dtype=datatype)
    hz = np.fromfunction(lambda i, j: (i * (j + 3)) / NX, (NX, NY), dtype=datatype)
    _fict_ = np.fromfunction(lambda i: i, (TMAX,), dtype=datatype)

    return TMAX, ex, ey, hz, _fict_


def get_impl(ctx):
    jit = ctx.jit
    np = ctx.numpy
    prange = ctx.prange

    @jit
    def kernel(TMAX, ex, ey, hz, _fict_):
        for t in range(TMAX):
            ey[0, :] = _fict_[t]
            ey[1:, :] -= 0.5 * (hz[1:, :] - hz[:-1, :])
            ex[:, 1:] -= 0.5 * (hz[:, 1:] - hz[:, :-1])
            hz[:-1, :-1] -= 0.7 * (
                ex[:-1, 1:] - ex[:-1, :-1] + ey[1:, :-1] - ey[:-1, :-1]
            )

    def wrapper(TMAX, ex, ey, hz, _fict_):
        kernel(TMAX, ex, ey, hz, _fict_)
        return ex, ey, hz

    return wrapper
