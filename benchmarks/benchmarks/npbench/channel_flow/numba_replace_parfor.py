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

from .common import initialize, get_impl, parameters, presets
import numba_mlir.mlir.benchmarking
from numba_mlir.mlir.benchmarking import (
    get_numba_replace_parfor_context,
    get_numpy_context,
    assert_allclose_recursive,
)


class Benchmark(numba_mlir.mlir.benchmarking.BenchmarkBase):
    params = presets
    param_names = ["preset"]

    def get_func(self):
        return get_impl(get_numba_replace_parfor_context())

    def initialize(self, preset):
        if self.is_validate:
            # skip in validation mode as it randomly fails with timeout
            raise NotImplementedError
        self.is_expected_failure = True
        preset = parameters[preset]
        return initialize(**preset)

    def validate(self, args, res):
        np_ver = get_impl(get_numpy_context())
        np_res = np_ver(*args)
        assert_allclose_recursive(res, np_res)
