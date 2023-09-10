# SPDX-FileCopyrightText: 2019 Serge Guelton
# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause


from numba_mlir.mlir.benchmarking import parse_config, filter_presets
from os import path

config = parse_config(path.join(path.dirname(__file__), "config.toml"))
parameters = dict(config["benchmark"]["parameters"])
presets = filter_presets(parameters.keys())


def initialize(N):
    from numpy.random import default_rng

    rng = default_rng(42)
    t0, p0, t1, p1 = (
        rng.random((N,)),
        rng.random((N,)),
        rng.random((N,)),
        rng.random((N,)),
    )
    return t0, p0, t1, p1


def get_impl(ctx):
    jit = ctx.jit
    np = ctx.numpy
    prange = ctx.prange

    def arc_distance(theta_1, phi_1, theta_2, phi_2):
        """
        Calculates the pairwise arc distance between all points in vector a and b.
        """
        temp = (
            np.sin((theta_2 - theta_1) / 2) ** 2
            + np.cos(theta_1) * np.cos(theta_2) * np.sin((phi_2 - phi_1) / 2) ** 2
        )
        distance_matrix = 2 * (np.arctan2(np.sqrt(temp), np.sqrt(1 - temp)))
        return distance_matrix

    return jit(arc_distance)
