# SPDX-FileCopyrightText: 2014 Jérôme Kieffer et al.
# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Jérôme Kieffer and Giannis Ashiotis. Pyfai: a python library for
high performance azimuthal integration on gpu, 2014. In Proceedings of the
7th European Conference on Python in Science (EuroSciPy 2014).
"""

from .azimint_naive import initialize, get_impl, parameters
import numba_mlir.mlir.benchmarking
from numba_mlir.mlir.benchmarking import get_numpy_context


class Benchmark(numba_mlir.mlir.benchmarking.BenchmarkBase):
    params = ["S"]
    param_names = ["preset"]

    def get_func(self, preset):
        return get_impl(get_numpy_context())

    def initialize(self, preset):
        preset = parameters[preset]
        N = preset["N"]
        npt = preset["npt"]
        return initialize(N, npt)

    def validate(self, args, res):
        # Assume numpy impl is valid
        pass
