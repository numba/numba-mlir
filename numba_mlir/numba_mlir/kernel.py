# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .mlir.kernel_impl import (
    kernel,
    get_global_id,
    get_local_id,
    get_group_id,
    get_global_size,
    get_local_size,
    atomic,
    kernel_func,
    DEFAULT_LOCAL_SIZE,
    barrier,
    mem_fence,
    CLK_LOCAL_MEM_FENCE,
    CLK_GLOBAL_MEM_FENCE,
    local,
    private,
    group,
)
from .mlir.kernel_sim import kernel as kernel_sim
