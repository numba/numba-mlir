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


def initialize(NR, NM, slab_per_bc, num_int_pts):
    from numpy.random import default_rng

    rng = default_rng(42)
    Ham = rng_complex((slab_per_bc + 1, NR, NR), rng)
    int_pts = rng_complex((num_int_pts,), rng)
    Y = rng_complex((NR, NM), rng)
    return NR, NM, slab_per_bc, Ham, int_pts, Y


def get_impl(ctx):
    jit = ctx.jit
    np = ctx.numpy
    prange = ctx.prange

    def contour_integral(NR, NM, slab_per_bc, Ham, int_pts, Y):
        P0 = np.zeros((NR, NM), dtype=np.complex128)
        P1 = np.zeros((NR, NM), dtype=np.complex128)
        for z in int_pts:
            Tz = np.zeros((NR, NR), dtype=np.complex128)
            for n in range(slab_per_bc + 1):
                zz = np.power(z, slab_per_bc / 2 - n)
                Tz += zz * Ham[n]
            if NR == NM:
                X = np.linalg.inv(Tz)
            else:
                X = np.linalg.solve(Tz, Y)
            if abs(z) < 1.0:
                X = -X
            P0 += X
            P1 += z * X

        return P0, P1

    return jit(contour_integral)
