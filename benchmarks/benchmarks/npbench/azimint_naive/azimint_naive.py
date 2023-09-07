# SPDX-FileCopyrightText: 2014 Jérôme Kieffer et al.
# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

from numba_mlir.mlir.benchmarking import parse_config, filter_presets
from os import path

config = parse_config(path.join(path.dirname(__file__), "azimint_naive.toml"))
parameters = config["benchmark"]["parameters"]
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

    def azimint_naive(data, radius, npt):
        rmax = radius.max()
        res = np.zeros(npt, dtype=np.float64)
        for i in prange(npt):
            r1 = rmax * i / npt
            r2 = rmax * (i + 1) / npt
            mask_r12 = np.logical_and((r1 <= radius), (radius < r2))
            values_r12 = data[mask_r12]
            res[i] = values_r12.mean()
        return res

    return jit(azimint_naive)
