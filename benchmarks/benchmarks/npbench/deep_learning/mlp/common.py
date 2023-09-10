# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

from numba_mlir.mlir.benchmarking import parse_config, filter_presets
from os import path

config = parse_config(path.join(path.dirname(__file__), "config.toml"))
parameters = dict(config["benchmark"]["parameters"])
presets = filter_presets(parameters.keys())


def initialize(C_in, N, S0, S1, S2):
    import numpy as np
    from numpy.random import default_rng

    rng = default_rng(42)

    mlp_sizes = [S0, S1, S2]  # [300, 100, 10]
    # Inputs
    input = np.random.rand(N, C_in).astype(np.float32)
    # Weights
    w1 = rng.random((C_in, mlp_sizes[0]), dtype=np.float32)
    b1 = rng.random((mlp_sizes[0],), dtype=np.float32)
    w2 = rng.random((mlp_sizes[0], mlp_sizes[1]), dtype=np.float32)
    b2 = rng.random((mlp_sizes[1],), dtype=np.float32)
    w3 = rng.random((mlp_sizes[1], mlp_sizes[2]), dtype=np.float32)
    b3 = rng.random((mlp_sizes[2],), dtype=np.float32)

    return input, w1, b1, w2, b2, w3, b3


def get_impl(ctx):
    jit = ctx.jit
    np = ctx.numpy
    prange = ctx.prange

    @jit
    def relu(x):
        return np.maximum(x, 0)

    # Numerically-stable version of softmax
    @jit
    def softmax(x):
        tmp_max = np.max(x, axis=-1, keepdims=True)
        tmp_out = np.exp(x - tmp_max)
        tmp_sum = np.sum(tmp_out, axis=-1, keepdims=True)
        return tmp_out / tmp_sum

    # 3-layer MLP
    @jit
    def mlp(input, w1, b1, w2, b2, w3, b3):
        x = relu(input @ w1 + b1)
        x = relu(x @ w2 + b2)
        x = softmax(x @ w3 + b3)  # Softmax call can be omitted if necessary
        return x

    return mlp


def get_impl_numba(ctx):
    jit = ctx.jit
    np = ctx.numpy
    prange = ctx.prange

    @jit
    def relu(x):
        return np.maximum(x, 0)

    # Numerically-stable version of softmax
    @jit
    def softmax(x):
        new_shape = (x.shape[0], 1)
        # tmp_max = np.max(x, axis=-1, keepdims=True)
        tmp_max = np.empty(new_shape, dtype=x.dtype)
        for i in prange(x.shape[0]):
            tmp_max[i, 0] = np.max(x[i])
        tmp_out = np.exp(x - tmp_max)
        # tmp_sum = np.sum(tmp_out, axis=-1, keepdims=True)
        tmp_sum = np.reshape(np.sum(tmp_out, axis=-1), new_shape)
        return tmp_out / tmp_sum

    # 3-layer MLP
    @jit
    def mlp(input, w1, b1, w2, b2, w3, b3):
        x = relu(input @ w1 + b1)
        x = relu(x @ w2 + b2)
        x = softmax(x @ w3 + b3)  # Softmax call can be omitted if necessary
        return x

    return mlp
