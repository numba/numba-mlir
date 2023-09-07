# SPDX-FileCopyrightText: 2014 Jérôme Kieffer et al.
# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

from numba_mlir.mlir.benchmarking import parse_config, filter_presets
from os import path

config = parse_config(path.join(path.dirname(__file__), "config.toml"))
parameters = dict(config["benchmark"]["parameters"])
presets = filter_presets(parameters.keys())


def initialize(N, npt):
    from numpy.random import default_rng

    rng = default_rng(42)
    data, radius = rng.random((N,)), rng.random((N,))
    return data, radius, npt


def get_impl(ctx):
    jit = ctx.jit
    np = ctx.numpy
    prange = ctx.prange

    def azimint_hist(data, radius, npt):
        histu = np.histogram(radius, npt)[0]
        histw = np.histogram(radius, npt, weights=data)[0]
        return histw / histu


    return jit(azimint_hist)


def get_impl_prange(ctx):
    jit = ctx.jit
    np = ctx.numpy
    prange = ctx.prange

    @jit
    def get_bin_edges_prange(a, bins):
        bin_edges = np.zeros((bins + 1,), dtype=np.float64)
        a_min = a.min()
        a_max = a.max()
        delta = (a_max - a_min) / bins
        for i in prange(bin_edges.shape[0]):
            bin_edges[i] = a_min + i * delta

        bin_edges[-1] = a_max  # Avoid roundoff error on last point
        return bin_edges


    @jit
    def compute_bin(x, bin_edges):
        # assuming uniform bins for now
        n = bin_edges.shape[0] - 1
        a_min = bin_edges[0]
        a_max = bin_edges[-1]

        # special case to mirror NumPy behavior for last bin
        if x == a_max:
            return n - 1  # a_max always in last bin

        return int(n * (x - a_min) / (a_max - a_min))


    @jit
    def histogram_prange(a, bins, weights):
        hist = np.zeros((bins,), dtype=a.dtype)
        bin_edges = get_bin_edges_prange(a, bins)

        for i in prange(a.shape[0]):
            bin = compute_bin(a[i], bin_edges)
            hist[bin] += weights[i]

        return hist, bin_edges


    @jit
    def azimint_hist(data, radius, npt):
        histu = np.histogram(radius, npt)[0]
        # histw = np.histogram(radius, npt, weights=data)[0]
        histw = histogram_prange(radius, npt, weights=data)[0]
        return histw / histu

    return azimint_hist
