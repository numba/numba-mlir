# SPDX-FileCopyrightText: 2018 Øystein Sture
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
    import numpy as np
    from numpy.random import default_rng

    rng = default_rng(42)
    data = rng.integers(0, 256, size=(N,), dtype=np.uint8)
    return (data,)


def get_impl(ctx):
    jit = ctx.jit
    np = ctx.numpy
    prange = ctx.prange

    # Adapted from https://gist.github.com/oysstu/68072c44c02879a2abf94ef350d1c7c6
    def crc16(data, poly=0x8408):
        """
        CRC-16-CCITT Algorithm
        """
        crc = 0xFFFF
        for b in data:
            cur_byte = 0xFF & b
            for _ in range(0, 8):
                if (crc & 0x0001) ^ (cur_byte & 0x0001):
                    crc = (crc >> 1) ^ poly
                else:
                    crc >>= 1
                cur_byte >>= 1
        crc = ~crc & 0xFFFF
        crc = (crc << 8) | ((crc >> 8) & 0xFF)

        return crc & 0xFFFF

    return jit(crc16)
