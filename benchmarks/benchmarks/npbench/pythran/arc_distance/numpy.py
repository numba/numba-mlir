# SPDX-FileCopyrightText: 2019 Serge Guelton
# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause


from .common import initialize, get_impl, parameters, presets
import numba_mlir.mlir.benchmarking
from numba_mlir.mlir.benchmarking import get_numpy_context


class Benchmark(numba_mlir.mlir.benchmarking.BenchmarkBase):
    params = presets
    param_names = ["preset"]

    def get_func(self):
        return get_impl(get_numpy_context())

    def initialize(self, preset):
        preset = parameters[preset]
        return initialize(**preset)

    def validate(self, args, res):
        # Assume numpy impl is valid
        pass
