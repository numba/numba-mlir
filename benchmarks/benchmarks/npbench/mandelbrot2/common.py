# SPDX-FileCopyrightText: 2017 Nicolas P. Rougier
# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

# more information at https://github.com/rougier/numpy-book

from numba_mlir.mlir.benchmarking import parse_config, filter_presets
from os import path

config = parse_config(path.join(path.dirname(__file__), "config.toml"))
parameters = dict(config["benchmark"]["parameters"])
presets = filter_presets(parameters.keys())


def initialize(xmin, xmax, ymin, ymax, xn, yn, itermax, horizon):
    # No initialization needed
    return xmin, xmax, ymin, ymax, xn, yn, itermax, horizon


def get_impl(ctx):
    jit = ctx.jit
    np = ctx.numpy
    prange = ctx.prange

    def mandelbrot(xmin, xmax, ymin, ymax, xn, yn, itermax, horizon=2.0):
        Xi, Yi = np.mgrid[0:xn, 0:yn]
        X = np.linspace(xmin, xmax, xn, dtype=np.float64)[Xi]
        Y = np.linspace(ymin, ymax, yn, dtype=np.float64)[Yi]
        C = X + Y * 1j
        N_ = np.zeros(C.shape, dtype=np.int64)
        Z_ = np.zeros(C.shape, dtype=np.complex128)
        Xi.shape = Yi.shape = C.shape = xn * yn

        Z = np.zeros(C.shape, np.complex128)
        for i in range(itermax):
            if not len(Z):
                break

            # Compute for relevant points only
            np.multiply(Z, Z, Z)
            np.add(Z, C, Z)

            # Failed convergence
            I = abs(Z) > horizon  # noqa: E741 math variable
            N_[Xi[I], Yi[I]] = i + 1
            Z_[Xi[I], Yi[I]] = Z[I]

            # Keep going with those who have not diverged yet
            np.logical_not(I, I)  # np.negative(I, I) not working any longer
            Z = Z[I]
            Xi, Yi = Xi[I], Yi[I]
            C = C[I]
        return Z_.T, N_.T

    return jit(mandelbrot)


def get_impl_numba(ctx):
    jit = ctx.jit
    np = ctx.numpy
    prange = ctx.prange

    @jit
    def mgrid(xn, yn):
        Xi = np.empty((xn, yn), dtype=np.int64)
        Yi = np.empty((xn, yn), dtype=np.int64)
        for i in range(xn):
            Xi[i, :] = i
        for j in range(yn):
            Yi[:, j] = j
        return Xi, Yi

    @jit
    def linspace(start, stop, num, dtype):
        X = np.empty((num,), dtype=dtype)
        dist = (stop - start) / (num - 1)
        for i in range(num):
            X[i] = start + i * dist
        return X

    @jit
    def mandelbrot(xmin, xmax, ymin, ymax, xn, yn, itermax, horizon=2.0):
        # Adapted from
        # https://thesamovar.wordpress.com/2009/03/22/fast-fractals-with-python-and-numpy/
        # Xi, Yi = np.mgrid[0:xn, 0:yn]
        # X = np.linspace(xmin, xmax, xn, dtype=np.float64)[Xi]
        # Y = np.linspace(ymin, ymax, yn, dtype=np.float64)[Yi]
        Xi, Yi = mgrid(xn, yn)
        X = linspace(xmin, xmax, xn, dtype=np.float64)
        Y = linspace(ymin, ymax, yn, dtype=np.float64)
        # C = X + Y*1j
        C = np.reshape(X, (xn, 1)) + Y * 1j
        N_ = np.zeros(C.shape, dtype=np.int64)
        Z_ = np.zeros(C.shape, dtype=np.complex128)
        # Xi.shape = Yi.shape = C.shape = xn*yn
        Xi = np.reshape(Xi, (xn * yn))
        Yi = np.reshape(Yi, (xn * yn))
        C = np.reshape(C, (xn * yn))

        Z = np.zeros(C.shape, np.complex128)
        for i in range(itermax):
            if not len(Z):
                break

            # Compute for relevant points only
            np.multiply(Z, Z, Z)
            np.add(Z, C, Z)

            # Failed convergence
            # I = abs(Z) > horizon
            I = np.absolute(Z) > horizon  # noqa: E741 math variable
            # N_[Xi[I], Yi[I]] = i+1
            for j in range(I.shape[0]):
                if I[j]:
                    N_[Xi[j], Yi[j]] = i + 1
            # Z_[Xi[I], Yi[I]] = Z[I]
            for j in range(I.shape[0]):
                if I[j]:
                    Z_[Xi[j], Yi[j]] = Z[j]

            # Keep going with those who have not diverged yet
            np.logical_not(I, I)  # np.negative(I, I) not working any longer
            Z = Z[I]
            Xi, Yi = Xi[I], Yi[I]
            C = C[I]
        return Z_.T, N_.T

    return mandelbrot
