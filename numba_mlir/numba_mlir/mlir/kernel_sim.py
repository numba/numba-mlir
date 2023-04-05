# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np

from . import kernel_sim_impl

from .kernel_base import KernelBase
from .kernel_impl import (
    get_global_id,
    get_local_id,
    get_group_id,
    get_global_size,
    get_local_size,
    atomic,
    atomic_add,
    atomic_sub,
    barrier,
    mem_fence,
    local,
    private,
    group,
)


class atomic_proxy:
    @staticmethod
    def add(arr, ind, val):
        new_val = arr[ind] + val
        arr[ind] = new_val
        return new_val

    @staticmethod
    def sub(arr, ind, val):
        new_val = arr[ind] - val
        arr[ind] = new_val
        return new_val


def mem_fence_proxy(flags):
    pass  # Nothing


class local_proxy:
    @staticmethod
    def array(shape, dtype):
        return kernel_sim_impl.local_array(shape, dtype)


class private_proxy:
    @staticmethod
    def array(shape, dtype):
        return kernel_sim_impl.private_array(shape, dtype)


class group_proxy:
    @staticmethod
    def reduce_add(value):
        return kernel_sim_impl.group_reduce(value, lambda a, b: a + b)


def barrier_proxy(flags):
    kernel_sim_impl.barrier()


_globals_to_replace = [
    (get_global_id, kernel_sim_impl.get_global_id),
    (get_local_id, kernel_sim_impl.get_local_id),
    (get_group_id, kernel_sim_impl.get_group_id),
    (get_global_size, kernel_sim_impl.get_global_size),
    (get_local_size, kernel_sim_impl.get_local_size),
    (atomic, atomic_proxy),
    (atomic_add, atomic_proxy.add),
    (atomic_sub, atomic_proxy.sub),
    (barrier, barrier_proxy),
    (mem_fence, mem_fence_proxy),
    (local, local_proxy),
    (local.array, local_proxy.array),
    (private, private_proxy),
    (private.array, private_proxy.array),
    (group, group_proxy),
    (group.reduce_add, group_proxy.reduce_add),
]

_barrier_ops = [barrier, group, group.reduce_add]


def _have_barrier_ops(func):
    for v in func.__globals__.values():
        for b in _barrier_ops:
            if v is b:
                return True
    return False


def _replace_global_func(global_obj):
    for old_val, new_val in _globals_to_replace:
        if global_obj is old_val:
            return new_val

    return global_obj


class Kernel(KernelBase):
    def __init__(self, func):
        super().__init__(func)

    def __call__(self, *args, **kwargs):
        self.check_call_args(args, kwargs)

        need_barrier = _have_barrier_ops(self.py_func)
        kernel_sim_impl.execute_kernel(
            self.global_size,
            self.local_size,
            self.py_func,
            args,
            need_barrier=need_barrier,
            replace_global_func=_replace_global_func,
        )


def kernel(func):
    return Kernel(func)
