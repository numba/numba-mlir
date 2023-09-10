# SPDX-FileCopyrightText: 2014-2021 ETH Zurich
# SPDX-FileCopyrightText: 2007 Free Software Foundation, Inc. <https://fsf.org/>
# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: GPL-3.0-or-later

from numba_mlir.mlir.benchmarking import parse_config, filter_presets
from os import path

config = parse_config(path.join(path.dirname(__file__), "config.toml"))
parameters = dict(config["benchmark"]["parameters"])
presets = filter_presets(parameters.keys())


def initialize(I, J, K):  # noqa: E741 math variable
    import numpy as np
    from numpy.random import default_rng

    rng = default_rng(42)

    # Define arrays
    in_field = rng.random((I + 4, J + 4, K))
    out_field = rng.random((I, J, K))
    coeff = rng.random((I, J, K))

    return in_field, out_field, coeff


def get_impl(ctx):
    jit = ctx.jit
    np = ctx.numpy
    prange = ctx.prange

    # Adapted from https://github.com/GridTools/gt4py/blob/1caca893034a18d5df1522ed251486659f846589/tests/test_integration/stencil_definitions.py#L194
    @jit
    def hdiff(in_field, out_field, coeff):
        I, J, K = (  # noqa: E741,F841 math variable, TODO: K is not used
            out_field.shape[0],
            out_field.shape[1],
            out_field.shape[2],
        )
        lap_field = 4.0 * in_field[1 : I + 3, 1 : J + 3, :] - (
            in_field[2 : I + 4, 1 : J + 3, :]
            + in_field[0 : I + 2, 1 : J + 3, :]
            + in_field[1 : I + 3, 2 : J + 4, :]
            + in_field[1 : I + 3, 0 : J + 2, :]
        )

        res = lap_field[1:, 1 : J + 1, :] - lap_field[:-1, 1 : J + 1, :]
        flx_field = np.where(
            (
                res
                * (
                    in_field[2 : I + 3, 2 : J + 2, :]
                    - in_field[1 : I + 2, 2 : J + 2, :]
                )
            )
            > 0,
            0,
            res,
        )

        res = lap_field[1 : I + 1, 1:, :] - lap_field[1 : I + 1, :-1, :]
        fly_field = np.where(
            (
                res
                * (
                    in_field[2 : I + 2, 2 : J + 3, :]
                    - in_field[2 : I + 2, 1 : J + 2, :]
                )
            )
            > 0,
            0,
            res,
        )

        out_field[:, :, :] = in_field[2 : I + 2, 2 : J + 2, :] - coeff[:, :, :] * (
            flx_field[1:, :, :]
            - flx_field[:-1, :, :]
            + fly_field[:, 1:, :]
            - fly_field[:, :-1, :]
        )

    def wrapper(in_field, out_field, coeff):
        hdiff(in_field, out_field, coeff)
        return out_field

    return wrapper
