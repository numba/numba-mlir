# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

from numba_mlir.mlir.benchmarking import parse_config, filter_presets
from os import path

config = parse_config(path.join(path.dirname(__file__), "config.toml"))
parameters = dict(config["benchmark"]["parameters"])
presets = filter_presets(parameters.keys())


def initialize(N, H, SM):
    import numpy as np
    from numpy.random import default_rng

    rng = default_rng(42)
    x = rng.random((N, H, SM, SM), dtype=np.float32)
    return (x,)


def get_impl(ctx):
    jit = ctx.jit
    np = ctx.numpy
    prange = ctx.prange

    # Numerically-stable version of softmax
    @jit
    def softmax(x):
        tmp_max = np.max(x, axis=-1, keepdims=True)
        tmp_out = np.exp(x - tmp_max)
        tmp_sum = np.sum(tmp_out, axis=-1, keepdims=True)
        return tmp_out / tmp_sum

    return softmax


def get_impl_numba(ctx):
    jit = ctx.jit
    np = ctx.numpy
    prange = ctx.prange

    # Numerically-stable version of softmax
    @jit
    def softmax(x):
        new_shape = (x.shape[0], x.shape[1], x.shape[2], 1)
        # tmp_max = np.max(x, axis=-1, keepdims=True)
        tmp_max = np.empty(new_shape, dtype=x.dtype)
        for i in prange(x.shape[3]):
            tmp_max[:, :, :, 0] = np.max(x[:, :, :, i])
        tmp_out = np.exp(x - tmp_max)
        # tmp_sum = np.sum(tmp_out, axis=-1, keepdims=True)
        tmp_sum = np.reshape(np.sum(tmp_out, axis=-1), new_shape)
        return tmp_out / tmp_sum

    return softmax
