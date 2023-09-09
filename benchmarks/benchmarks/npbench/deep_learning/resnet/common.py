# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

from numba_mlir.mlir.benchmarking import parse_config, filter_presets
from os import path

config = parse_config(path.join(path.dirname(__file__), "config.toml"))
parameters = dict(config["benchmark"]["parameters"])
presets = filter_presets(parameters.keys())


def initialize(N, W, H, C1, C2):
    import numpy as np
    from numpy.random import default_rng

    rng = default_rng(42)

    # Input
    input = rng.random((N, H, W, C1), dtype=np.float32)
    # Weights
    conv1 = rng.random((1, 1, C1, C2), dtype=np.float32)
    conv2 = rng.random((3, 3, C2, C2), dtype=np.float32)
    conv3 = rng.random((1, 1, C2, C1), dtype=np.float32)
    return (input, conv1, conv2, conv3)


def get_impl(ctx):
    jit = ctx.jit
    np = ctx.numpy
    prange = ctx.prange

    def relu(x):
        return np.maximum(x, 0)

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

    # Batch normalization operator, as used in ResNet
    @jit
    def batchnorm2d(x, eps=1e-5):
        mean = np.mean(x, axis=0, keepdims=True)
        std = np.std(x, axis=0, keepdims=True)
        return (x - mean) / np.sqrt(std + eps)

    # Bottleneck residual block (after initial convolution, without downsampling)
    # in the ResNet-50 CNN (inference)
    @jit
    def resnet_basicblock(input, conv1, conv2, conv3):
        # Pad output of first convolution for second convolution
        padded = np.zeros(
            (input.shape[0], input.shape[1] + 2, input.shape[2] + 2, conv1.shape[3])
        )

        padded[:, 1:-1, 1:-1, :] = conv2d(input, conv1)
        x = batchnorm2d(padded)
        x = relu(x)

        x = conv2d(x, conv2)
        x = batchnorm2d(x)
        x = relu(x)
        x = conv2d(x, conv3)
        x = batchnorm2d(x)
        return relu(x + input)

    return resnet_basicblock


def get_impl_numba(ctx):
    jit = ctx.jit
    np = ctx.numpy
    prange = ctx.prange

    @jit
    def relu(x):
        return np.maximum(x, 0)

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

    # Batch normalization operator, as used in ResNet
    @jit
    def batchnorm2d(x, eps=1e-5):
        # mean = np.mean(x, axis=0, keepdims=True)
        mean = np.empty(x.shape, dtype=x.dtype)
        mean[:] = np.sum(x, axis=0) / x.shape[0]
        # std = np.std(x, axis=0, keepdims=True)
        std = np.empty(x.shape, dtype=x.dtype)
        std[:] = np.sqrt(np.sum((x - mean) ** 2, axis=0) / x.shape[0])
        return (x - mean) / np.sqrt(std + eps)

    # Bottleneck residual block (after initial convolution, without downsampling)
    # in the ResNet-50 CNN (inference)
    @jit
    def resnet_basicblock(input, conv1, conv2, conv3):
        # Pad output of first convolution for second convolution
        padded = np.zeros(
            (input.shape[0], input.shape[1] + 2, input.shape[2] + 2, conv1.shape[3])
        )

        padded[:, 1:-1, 1:-1, :] = conv2d(input, conv1)
        x = batchnorm2d(padded)
        x = relu(x)

        x = conv2d(x, conv2)
        x = batchnorm2d(x)
        x = relu(x)
        x = conv2d(x, conv3)
        x = batchnorm2d(x)
        return relu(x + input)

    return resnet_basicblock
