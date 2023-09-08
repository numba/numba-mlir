# SPDX-FileCopyrightText: 2017 Lorena A. Barba, Gilbert F. Forsyth.
# SPDX-FileCopyrightText: 2018 Barba, Lorena A., and Forsyth, Gilbert F.
# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

"""
CFD Python: the 12 steps to Navier-Stokes equations.
Journal of Open Source Education, 1(9), 21,
https://doi.org/10.21105/jose.00021
"""

from numba_mlir.mlir.benchmarking import parse_config, filter_presets
from os import path

config = parse_config(path.join(path.dirname(__file__), "config.toml"))
parameters = dict(config["benchmark"]["parameters"])
presets = filter_presets(parameters.keys())


def initialize(ny, nx, nt, nit, rho, nu):
    import numpy as np

    u = np.zeros((ny, nx), dtype=np.float64)
    v = np.zeros((ny, nx), dtype=np.float64)
    p = np.zeros((ny, nx), dtype=np.float64)
    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)
    dt = 0.1 / ((nx - 1) * (ny - 1))
    return ny, nx, nt, nit, u, v, dt, dx, dy, p, rho, nu


def get_impl(ctx):
    jit = ctx.jit
    np = ctx.numpy
    prange = ctx.prange

    @jit
    def build_up_b(b, rho, dt, u, v, dx, dy):
        b[1:-1, 1:-1] = rho * (
            1
            / dt
            * (
                (u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx)
                + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)
            )
            - ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx)) ** 2
            - 2
            * (
                (u[2:, 1:-1] - u[0:-2, 1:-1])
                / (2 * dy)
                * (v[1:-1, 2:] - v[1:-1, 0:-2])
                / (2 * dx)
            )
            - ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) ** 2
        )

    @jit
    def pressure_poisson(nit, p, dx, dy, b):
        pn = np.empty_like(p)
        pn = p.copy()

        for q in range(nit):
            pn = p.copy()
            p[1:-1, 1:-1] = (
                (pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2
                + (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2
            ) / (2 * (dx**2 + dy**2)) - dx**2 * dy**2 / (
                2 * (dx**2 + dy**2)
            ) * b[
                1:-1, 1:-1
            ]

            p[:, -1] = p[:, -2]  # dp/dx = 0 at x = 2
            p[0, :] = p[1, :]  # dp/dy = 0 at y = 0
            p[:, 0] = p[:, 1]  # dp/dx = 0 at x = 0
            p[-1, :] = 0  # p = 0 at y = 2

    @jit
    def cavity_flow(nx, ny, nt, nit, u, v, dt, dx, dy, p, rho, nu):
        un = np.empty_like(u)
        vn = np.empty_like(v)
        b = np.zeros((ny, nx))

        for n in range(nt):
            un = u.copy()
            vn = v.copy()

            build_up_b(b, rho, dt, u, v, dx, dy)
            pressure_poisson(nit, p, dx, dy, b)

            u[1:-1, 1:-1] = (
                un[1:-1, 1:-1]
                - un[1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1] - un[1:-1, 0:-2])
                - vn[1:-1, 1:-1] * dt / dy * (un[1:-1, 1:-1] - un[0:-2, 1:-1])
                - dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2])
                + nu
                * (
                    dt / dx**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2])
                    + dt
                    / dy**2
                    * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])
                )
            )

            v[1:-1, 1:-1] = (
                vn[1:-1, 1:-1]
                - un[1:-1, 1:-1] * dt / dx * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2])
                - vn[1:-1, 1:-1] * dt / dy * (vn[1:-1, 1:-1] - vn[0:-2, 1:-1])
                - dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1])
                + nu
                * (
                    dt / dx**2 * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2])
                    + dt
                    / dy**2
                    * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])
                )
            )

            u[0, :] = 0
            u[:, 0] = 0
            u[:, -1] = 0
            u[-1, :] = 1  # set velocity on cavity lid equal to 1
            v[0, :] = 0
            v[-1, :] = 0
            v[:, 0] = 0
            v[:, -1] = 0

    def wrapper(nx, ny, nt, nit, u, v, dt, dx, dy, p, rho, nu):
        cavity_flow(nx, ny, nt, nit, u, v, dt, dx, dy, p, rho, nu)
        return u, v, p, dx, dy, dt

    return wrapper
