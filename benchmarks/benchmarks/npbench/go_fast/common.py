# SPDX-FileCopyrightText: 2012-2020 Anaconda, Inc. and others
# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

# https://numba.readthedocs.io/en/stable/user/5minguide.html

from numba_mlir.mlir.benchmarking import parse_config, filter_presets
from os import path

config = parse_config(path.join(path.dirname(__file__), "config.toml"))
parameters = dict(config["benchmark"]["parameters"])
presets = filter_presets(parameters.keys())


def initialize(N):
    import numpy as np
    from numpy.random import default_rng

    rng = default_rng(42)
    x = rng.random((N, N), dtype=np.float64)
    return (x,)


def get_impl(ctx):
    jit = ctx.jit
    np = ctx.numpy
    prange = ctx.prange

    def go_fast(a):
        trace = 0.0
        for i in prange(a.shape[0]):
            trace += np.tanh(a[i, i])
        return a + trace

    return jit(go_fast)
