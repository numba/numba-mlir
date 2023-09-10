# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

from numba_mlir.mlir.benchmarking import parse_config, filter_presets
from os import path

config = parse_config(path.join(path.dirname(__file__), "config.toml"))
parameters = dict(config["benchmark"]["parameters"])
presets = filter_presets(parameters.keys())


def rng_complex(shape, rng):
    return rng.random(shape) + rng.random(shape) * 1j


def initialize(R, K):
    import numpy as np
    from numpy.random import default_rng

    rng = default_rng(42)

    N = R**K
    X = rng_complex((N,), rng)
    Y = np.zeros_like(X, dtype=np.complex128)

    return N, R, K, X, Y


def get_impl(ctx):
    jit = ctx.jit
    np = ctx.numpy
    prange = ctx.prange

    @jit
    def stockham_fft(N, R, K, x, y):
        # Generate DFT matrix for radix R.
        # Define transient variable for matrix.
        i_coord, j_coord = np.mgrid[0:R, 0:R]
        dft_mat = np.empty((R, R), dtype=np.complex128)
        dft_mat = np.exp(-2.0j * np.pi * i_coord * j_coord / R)
        # Move input x to output y
        # to avoid overwriting the input.
        y[:] = x[:]

        ii_coord, jj_coord = np.mgrid[0:R, 0 : R**K]

        # Main Stockham loop
        for i in range(K):
            # Stride permutation
            yv = np.reshape(y, (R**i, R, R ** (K - i - 1)))
            tmp_perm = np.transpose(yv, axes=(1, 0, 2))
            # Twiddle Factor multiplication
            D = np.empty((R, R**i, R ** (K - i - 1)), dtype=np.complex128)
            tmp = np.exp(
                -2.0j
                * np.pi
                * ii_coord[:, : R**i]
                * jj_coord[:, : R**i]
                / R ** (i + 1)
            )
            D[:] = np.repeat(np.reshape(tmp, (R, R**i, 1)), R ** (K - i - 1), axis=2)
            tmp_twid = np.reshape(tmp_perm, (N,)) * np.reshape(D, (N,))
            # Product with Butterfly
            y[:] = np.reshape(dft_mat @ np.reshape(tmp_twid, (R, R ** (K - 1))), (N,))

    def wrapper(N, R, K, x, y):
        stockham_fft(N, R, K, x, y)
        return y

    return wrapper


def get_impl_numba(ctx):
    jit = ctx.jit
    np = ctx.numpy
    prange = ctx.prange

    @jit
    def mgrid(xn, yn):
        Xi = np.empty((xn, yn), dtype=np.uint32)
        Yi = np.empty((xn, yn), dtype=np.uint32)
        for i in prange(xn):
            Xi[i, :] = i
        for j in prange(yn):
            Yi[:, j] = j
        return Xi, Yi

    @jit
    def stockham_fft(N, R, K, x, y):
        # Generate DFT matrix for radix R.
        # Define transient variable for matrix.
        i_coord, j_coord = mgrid(R, R)
        dft_mat = np.empty((R, R), dtype=np.complex128)
        dft_mat = np.exp(-2.0j * np.pi * i_coord * j_coord / R)
        # Move input x to output y
        # to avoid overwriting the input.
        y[:] = x[:]

        # ii_coord, jj_coord = np.mgrid[0:R, 0:R**K]
        ii_coord, jj_coord = mgrid(R, R**K)

        # Main Stockham loop
        for i in range(K):
            # Stride permutation
            yv = np.reshape(y, (R**i, R, R ** (K - i - 1)))
            # tmp_perm = np.transpose(yv, axes=(1, 0, 2))
            tmp_perm = np.transpose(yv, axes=(1, 0, 2)).copy()
            # Twiddle Factor multiplication
            D = np.empty((R, R**i, R ** (K - i - 1)), dtype=np.complex128)
            tmp = np.exp(
                -2.0j
                * np.pi
                * ii_coord[:, : R**i]
                * jj_coord[:, : R**i]
                / R ** (i + 1)
            )
            # D[:] = np.repeat(np.reshape(tmp, (R, R**i, 1)), R ** (K-i-1), axis=2)
            for k in prange(R ** (K - i - 1)):
                D[:, :, k] = tmp
            tmp_twid = np.reshape(tmp_perm, (N,)) * np.reshape(D, (N,))
            # Product with Butterfly
            y[:] = np.reshape(dft_mat @ np.reshape(tmp_twid, (R, R ** (K - 1))), (N,))

    def wrapper(N, R, K, x, y):
        stockham_fft(N, R, K, x, y)
        return y

    return wrapper
