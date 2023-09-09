# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

from numba_mlir.mlir.benchmarking import parse_config, filter_presets
from os import path

config = parse_config(path.join(path.dirname(__file__), "config.toml"))
parameters = dict(config["benchmark"]["parameters"])
presets = filter_presets(parameters.keys())


def initialize(C_in, C_out, H, K, N, W):
    import numpy as np
    from numpy.random import default_rng

    rng = default_rng(42)
    # NHWC data layout
    input = rng.random((N, H, W, C_in), dtype=np.float32)
    # Weights
    weights = rng.random((K, K, C_in, C_out), dtype=np.float32)
    bias = rng.random((C_out,), dtype=np.float32)
    return input, weights, bias


def get_impl(ctx):
    jit = ctx.jit
    np = ctx.numpy
    prange = ctx.prange

    # Deep learning convolutional operator (stride = 1)
    @jit
    def conv2d(input, weights):
        K = weights.shape[0]  # Assuming square kernel
        N = input.shape[0]
        H_out = input.shape[1] - K + 1
        W_out = input.shape[2] - K + 1
        C_out = weights.shape[3]
        output = np.empty((N, H_out, W_out, C_out), dtype=np.float32)

        # Loop structure adapted from https://github.com/SkalskiP/ILearnDeepLearning.py/blob/ba0b5ba589d4e656141995e8d1a06d44db6ce58d/01_mysteries_of_neural_networks/06_numpy_convolutional_neural_net/src/layers/convolutional.py#L88
        for i in range(H_out):
            for j in range(W_out):
                output[:, i, j, :] = np.sum(
                    input[:, i : i + K, j : j + K, :, np.newaxis]
                    * weights[np.newaxis, :, :, :],
                    axis=(1, 2, 3),
                )

        return output

    @jit
    def conv2d_bias(input, weights, bias):
        return conv2d(input, weights) + bias

    return conv2d_bias


def get_impl_numba(ctx):
    jit = ctx.jit
    np = ctx.numpy
    prange = ctx.prange

    # Deep learning convolutional operator (stride = 1)
    @jit
    def conv2d(input, weights):
        K = weights.shape[0]  # Assuming square kernel
        N = input.shape[0]
        H_out = input.shape[1] - K + 1
        W_out = input.shape[2] - K + 1
        C_in = input.shape[3]
        C_out = weights.shape[3]
        output = np.empty((N, H_out, W_out, C_out), dtype=np.float32)

        # Loop structure adapted from https://github.com/SkalskiP/ILearnDeepLearning.py/blob/ba0b5ba589d4e656141995e8d1a06d44db6ce58d/01_mysteries_of_neural_networks/06_numpy_convolutional_neural_net/src/layers/convolutional.py#L88
        for i in prange(H_out):
            for j in prange(W_out):
                # output[:, i, j, :] = np.sum(
                #     input[:, i:i + K, j:j + K, :, np.newaxis] *
                #     weights[np.newaxis, :, :, :],
                #     axis=(1, 2, 3),
                # )
                # Reshape supported only on contiguous arrays
                inp = input[:, i : i + K, j : j + K, :].copy()
                # Tuple of ints not supported in axis keyword
                output[:, i, j, :] = np.sum(
                    np.sum(
                        np.sum(
                            np.reshape(inp, (N, K, K, C_in, 1))
                            * np.reshape(weights, (1, K, K, C_in, C_out)),
                            axis=1,
                        ),
                        axis=1,
                    ),
                    axis=1,
                )

        return output

    @jit
    def conv2d_bias(input, weights, bias):
        return conv2d(input, weights) + bias

    return conv2d_bias
