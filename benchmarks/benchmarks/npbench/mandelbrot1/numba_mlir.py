# SPDX-FileCopyrightText: 2017 Nicolas P. Rougier
# SPDX-FileCopyrightText: 2021 ETH Zurich and the NPBench authors
# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

# more information at https://github.com/rougier/numpy-book

from .common import initialize, get_impl, get_impl_numba, parameters, presets
import numba_mlir.mlir.benchmarking
from numba_mlir.mlir.benchmarking import (
    get_numba_mlir_context,
    get_numpy_context,
    assert_allclose_recursive,
    SkipNotImplemented,
)


class Benchmark(numba_mlir.mlir.benchmarking.BenchmarkBase):
    params = presets
    param_names = ["preset"]

    def get_func(self):
        return get_impl_numba(get_numba_mlir_context())

    def initialize(self, preset):
        raise SkipNotImplemented("nans")

        preset = parameters[preset]
        return initialize(**preset)

    def validate(self, args, res):
        np_ver = get_impl(get_numpy_context())
        np_res = np_ver(*args)
        assert_allclose_recursive(res, np_res, rtol=0.002)
