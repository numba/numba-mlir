# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from asv_runner.benchmarks.mark import skip_benchmark_if

from numba_mlir.kernel import kernel, get_global_id, DEFAULT_LOCAL_SIZE
from numba_mlir.mlir.benchmarking import (
    has_dpctl,
    get_dpctl_devices,
)

import numpy as np


def get_func(count):
    # TODO: something better
    if count == 1:

        def func(arg1):
            i = get_global_id(0)
            arg1[i] = i

        return func
    if count == 2:

        def func(arg1, arg2):
            i = get_global_id(0)
            arg1[i] = i

        return func
    if count == 6:

        def func(arg1, arg2, arg3, arg4, arg5, arg6):
            i = get_global_id(0)
            arg1[i] = i

        return func
    if count == 12:

        def func(
            arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12
        ):
            i = get_global_id(0)
            arg1[i] = i

        return func

    assert False


class KernelDispatcher:
    version = "base"

    params = [[1, 2, 6, 12], get_dpctl_devices()]
    # params = [[1], get_dpctl_devices()]
    param_names = ["count", "device"]

    @skip_benchmark_if(not has_dpctl())
    def setup(self, count, device):
        import dpctl.tensor as dpt

        array = dpt.empty(8, dtype=np.int32)

        func = kernel(get_func(count))
        self.base_func = func
        self.func = func[8, DEFAULT_LOCAL_SIZE]
        self.args = (array,) * count
        self.base_func[8, DEFAULT_LOCAL_SIZE](*self.args)
        self.func(*self.args)

    def time_dispatcher(self, count, device):
        self.func(*self.args)

    def time_full_dispatcher(self, count, device):
        self.base_func[8, DEFAULT_LOCAL_SIZE](*self.args)
