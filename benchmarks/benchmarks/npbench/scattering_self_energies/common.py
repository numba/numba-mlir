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


def initialize(Nkz, NE, Nqz, Nw, N3D, NA, NB, Norb):
    import numpy as np
    from numpy.random import default_rng

    rng = default_rng(42)

    neigh_idx = np.ndarray([NA, NB], dtype=np.int32)
    for i in range(NA):
        neigh_idx[i] = np.positive(np.arange(i - NB / 2, i + NB / 2) % NA)
    dH = rng_complex([NA, NB, N3D, Norb, Norb], rng)
    G = rng_complex([Nkz, NE, NA, Norb, Norb], rng)
    D = rng_complex([Nqz, Nw, NA, NB, N3D, N3D], rng)
    Sigma = np.zeros([Nkz, NE, NA, Norb, Norb], dtype=np.complex128)

    return neigh_idx, dH, G, D, Sigma


def get_impl(ctx):
    jit = ctx.jit
    np = ctx.numpy
    prange = ctx.prange

    @jit
    def scattering_self_energies(neigh_idx, dH, G, D, Sigma):
        for k in prange(G.shape[0]):
            for E in prange(G.shape[1]):
                for q in prange(D.shape[0]):
                    for w in prange(D.shape[1]):
                        for i in prange(D.shape[-2]):
                            for j in prange(D.shape[-1]):
                                for a in prange(neigh_idx.shape[0]):
                                    for b in prange(neigh_idx.shape[1]):
                                        if E - w >= 0:
                                            dHG = (
                                                G[k, E - w, neigh_idx[a, b]]
                                                @ dH[a, b, i]
                                            )
                                            dHD = dH[a, b, j] * D[q, w, a, b, i, j]
                                            Sigma[k, E, a] += dHG @ dHD

    def wrapper(neigh_idx, dH, G, D, Sigma):
        scattering_self_energies(neigh_idx, dH, G, D, Sigma)
        return Sigma

    return wrapper
