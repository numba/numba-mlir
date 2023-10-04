# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys

from numba import prange
from numba.core import types
from numba.core.typing.npydecl import parse_dtype, parse_shape
from numba.core.types.npytypes import Array
from numba.core.typing.templates import (
    AbstractTemplate,
    ConcreteTemplate,
    signature,
)

from .target import infer_global
from .linalg_builder import is_int, dtype_str, FuncRegistry, literal
from .numpy.funcs import register_func
from .func_registry import add_func

from ..decorators import mlir_njit
from .kernel_base import KernelBase
from .dpctl_interop import check_usm_ndarray_args

registry = FuncRegistry()

register_func = registry.register_func


def _stub_error():
    raise NotImplementedError("This is a stub")


class _gpu_range(object):
    def __new__(cls, *args):
        return range(*args)


add_func(_gpu_range, "_gpu_range")


@infer_global(_gpu_range, typing_key=_gpu_range)
class _RangeId(ConcreteTemplate):
    cases = [
        signature(types.range_state32_type, types.int32),
        signature(types.range_state32_type, types.int32, types.int32),
        signature(types.range_state32_type, types.int32, types.int32, types.int32),
        signature(types.range_state64_type, types.int64),
        signature(types.range_state64_type, types.int64, types.int64),
        signature(types.range_state64_type, types.int64, types.int64, types.int64),
        signature(types.unsigned_range_state64_type, types.uint64),
        signature(types.unsigned_range_state64_type, types.uint64, types.uint64),
        signature(
            types.unsigned_range_state64_type, types.uint64, types.uint64, types.uint64
        ),
    ]


def _set_default_local_size():
    _stub_error()


@registry.register_func("_gpu_range", _gpu_range)
def range_impl(builder, begin, end=None, step=1):
    end = literal(end)
    if end is None:
        end = begin
        begin = 0

    index = builder.index
    begin = builder.cast(begin, index)
    end = builder.cast(end, index)
    step = builder.cast(step, index)
    return (begin, end, step)


@registry.register_func("_set_default_local_size", _set_default_local_size)
def _set_default_local_size_impl(builder, *args):
    index_type = builder.index
    i64 = builder.int64
    zero = builder.cast(0, index_type)
    res = (zero, zero, zero)
    res = builder.external_call("set_default_local_size", inputs=args, outputs=res)
    return tuple(builder.cast(r, i64) for r in res)


@infer_global(_set_default_local_size)
class _SetDefaultLocalSizeId(ConcreteTemplate):
    cases = [
        signature(
            types.UniTuple(types.int64, 3), types.int64, types.int64, types.int64
        ),
    ]


def _kernel_body1(global_size, local_size, body, *args):
    x = global_size[0]
    lx = local_size[0]
    _set_default_local_size(lx, 1, 1)
    for gi in _gpu_range(x):
        for gj in _gpu_range(1):
            for gk in _gpu_range(1):
                body(*args)


def _kernel_body2(global_size, local_size, body, *args):
    x, y = global_size
    lx, ly = local_size
    _set_default_local_size(lx, ly, 1)
    for gi in _gpu_range(x):
        for gj in _gpu_range(y):
            for gk in _gpu_range(1):
                body(*args)


def _kernel_body3(global_size, local_size, body, *args):
    x, y, z = global_size
    lx, ly, lz = local_size
    _set_default_local_size(lx, ly, lz)
    for gi in _gpu_range(x):
        for gj in _gpu_range(y):
            for gk in _gpu_range(z):
                body(*args)


def _kernel_body_def_size1(global_size, body, *args):
    x = global_size[0]
    for gi in _gpu_range(x):
        for gj in _gpu_range(1):
            for gk in _gpu_range(1):
                body(*args)


def _kernel_body_def_size2(global_size, body, *args):
    x, y = global_size
    for gi in _gpu_range(x):
        for gj in _gpu_range(y):
            for gk in _gpu_range(1):
                body(*args)


def _kernel_body_def_size3(global_size, body, *args):
    x, y, z = global_size
    for gi in _gpu_range(x):
        for gj in _gpu_range(y):
            for gk in _gpu_range(z):
                body(*args)


def _decorate_kern_body(body, kwargs):
    return mlir_njit(enable_gpu_pipeline=True, **kwargs)(body)


class Kernel(KernelBase):
    def __init__(self, func, kwargs):
        super().__init__(func)
        fp64_truncate = kwargs.get("gpu_fp64_truncate", False)
        use_64bit_index = kwargs.get("gpu_use_64bit_index", True)
        self._jit_func = mlir_njit(
            inline="always",
            enable_gpu_pipeline=True,
            gpu_fp64_truncate=fp64_truncate,
            gpu_use_64bit_index=use_64bit_index,
        )(func)
        self._kern_body = (
            _decorate_kern_body(_kernel_body1, kwargs),
            _decorate_kern_body(_kernel_body2, kwargs),
            _decorate_kern_body(_kernel_body3, kwargs),
        )
        self._kern_body_def_size = (
            _decorate_kern_body(_kernel_body_def_size1, kwargs),
            _decorate_kern_body(_kernel_body_def_size2, kwargs),
            _decorate_kern_body(_kernel_body_def_size3, kwargs),
        )

    def __call__(self, *args, **kwargs):
        self.check_call_args(args, kwargs)

        # kwargs is not supported
        check_usm_ndarray_args(args)

        func_index = len(self.global_size) - 1
        assert (
            func_index >= 0 and func_index < 3
        ), f"Invalid dim count: {len(self.global_size)}"

        local_size = self.local_size
        if len(local_size) != 0:
            self._kern_body[func_index](
                self.global_size, self.local_size, self._jit_func, *args
            )
        else:
            self._kern_body_def_size[func_index](
                self.global_size, self._jit_func, *args
            )


def kernel(func=None, **kwargs):
    if func is None:

        def wrapper(f):
            return Kernel(f, kwargs)

        return wrapper
    return Kernel(func, kwargs)


DEFAULT_LOCAL_SIZE = ()

kernel_func = mlir_njit(inline="always")


def _define_api_funcs():
    kernel_api_funcs = [
        "get_global_id",
        "get_local_id",
        "get_group_id",
        "get_global_size",
        "get_local_size",
    ]

    def get_func(func_name):
        def api_func_impl(builder, axis):
            if isinstance(axis, int) or is_int(axis.type, builder):
                res = builder.cast(0, builder.int64)
                return builder.external_call(func_name, axis, res)

        return api_func_impl

    def get_stub_func(func_name):
        exec(f"def {func_name}(axis): _stub_error()")
        return eval(func_name)

    class ApiFuncId(ConcreteTemplate):
        cases = [signature(types.uint64, types.uint64)]

    this_module = sys.modules[__name__]

    for func_name in kernel_api_funcs:
        func = get_stub_func(func_name)
        setattr(this_module, func_name, func)

        infer_global(func)(ApiFuncId)
        registry.register_func(func_name, func)(get_func(func_name))


_define_api_funcs()
del _define_api_funcs


class Stub(object):
    """A stub object to represent special objects which is meaningless
    outside the context of DPPY compilation context.
    """

    __slots__ = ()  # don't allocate __dict__

    def __new__(cls):
        raise NotImplementedError("%s is not instantiable" % cls)


class atomic(Stub):
    pass


def _define_atomic_funcs():
    funcs = ["add", "sub"]

    def get_func(func_name, sub):
        if sub:

            def api_func_impl(builder, arr, idx, val):
                if not (isinstance(idx, int) and literal(idx) == 0):
                    arr = builder.subview(arr, idx)

                dtype = arr.dtype
                val = builder.cast(-val, dtype)
                fname = f"atomic_add_{dtype_str(builder, dtype)}_{len(arr.shape)}"
                return builder.external_call(fname, (arr, val), val)

        else:

            def api_func_impl(builder, arr, idx, val):
                if not (isinstance(idx, int) and literal(idx) == 0):
                    arr = builder.subview(arr, idx)

                dtype = arr.dtype
                val = builder.cast(val, dtype)
                fname = f"{func_name}_{dtype_str(builder, dtype)}_{len(arr.shape)}"
                return builder.external_call(fname, (arr, val), val)

        return api_func_impl

    def get_stub_func(func_name):
        exec(f"def {func_name}(arr, idx, val): _stub_error()")
        return eval(func_name)

    class _AtomicId(AbstractTemplate):
        def generic(self, args, kws):
            assert not kws
            ary, idx, val = args

            if ary.ndim == 1:
                return signature(ary.dtype, ary, types.intp, ary.dtype)
            elif ary.ndim > 1:
                return signature(ary.dtype, ary, idx, ary.dtype)

    this_module = sys.modules[__name__]

    for name in funcs:
        func_name = f"atomic_{name}"
        func = get_stub_func(func_name)
        setattr(this_module, func_name, func)

        infer_global(func)(_AtomicId)
        registry.register_func(func_name, func)(get_func(func_name, name == "sub"))
        setattr(atomic, name, func)


_define_atomic_funcs()
del _define_atomic_funcs


# mem fence
LOCAL_MEM_FENCE = 0x1
GLOBAL_MEM_FENCE = 0x2


def barrier(flags=None):
    _stub_error()


@registry.register_func("barrier", barrier)
def _barrier_impl(builder, flags=None):
    if flags is None:
        flags = GLOBAL_MEM_FENCE

    res = 0  # TODO: remove
    return builder.external_call("kernel_barrier", inputs=flags, outputs=res)


@infer_global(barrier)
class _BarrierId(ConcreteTemplate):
    cases = [signature(types.void, types.int64), signature(types.void)]


def mem_fence(flags=None):
    _stub_error()


@registry.register_func("mem_fence", mem_fence)
def _memf_fence_impl(builder, flags=None):
    if flags is None:
        flags = GLOBAL_MEM_FENCE

    res = 0  # TODO: remove
    return builder.external_call("kernel_mem_fence", inputs=flags, outputs=res)


@infer_global(mem_fence)
class _MemFenceId(ConcreteTemplate):
    cases = [signature(types.void, types.int64), signature(types.void)]


class local(Stub):
    pass


def local_array(shape, dtype):
    _stub_error()


setattr(local, "array", local_array)


@infer_global(local_array)
class _LocalId(AbstractTemplate):
    def generic(self, args, kws):
        shape = kws["shape"] if "shape" in kws else args[0]
        dtype = kws["dtype"] if "dtype" in kws else args[1]

        ndim = parse_shape(shape)
        dtype = parse_dtype(dtype)
        arr_type = Array(dtype=dtype, ndim=ndim, layout="C")
        return signature(arr_type, shape, dtype)


@registry.register_func("local_array", local_array)
def _local_array_impl(builder, shape, dtype):
    try:
        len(shape)  # will raise if not available
    except:
        shape = (shape,)

    func_name = f"local_array_{dtype_str(builder, dtype)}_{len(shape)}"
    res = builder.init_tensor(shape, dtype)
    return builder.external_call(
        func_name, inputs=shape, outputs=res, return_tensor=True
    )


class private(Stub):
    pass


def private_array(shape, dtype):
    _stub_error()


setattr(private, "array", private_array)


@infer_global(private_array)
class _PrivateId(AbstractTemplate):
    def generic(self, args, kws):
        shape = kws["shape"] if "shape" in kws else args[0]
        dtype = kws["dtype"] if "dtype" in kws else args[1]

        ndim = parse_shape(shape)
        dtype = parse_dtype(dtype)
        arr_type = Array(dtype=dtype, ndim=ndim, layout="C")
        return signature(arr_type, shape, dtype)


@registry.register_func("private_array", private_array)
def _private_array_impl(builder, shape, dtype):
    try:
        len(shape)  # will raise if not available
    except:
        shape = (shape,)

    func_name = f"private_array_{dtype_str(builder, dtype)}_{len(shape)}"
    res = builder.init_tensor(shape, dtype)
    return builder.external_call(
        func_name, inputs=shape, outputs=res, return_tensor=True
    )


class group(Stub):
    pass


def _define_group_funcs():
    ops = ["add", "mul", "min", "max"]

    def get_func(func_name):
        def api_func_impl(builder, value):
            elem_type = value.type
            api_func_name = f"{func_name}_{dtype_str(builder, elem_type)}"
            res = builder.cast(0, elem_type)
            return builder.external_call(api_func_name, inputs=value, outputs=res)

        return api_func_impl

    def get_stub_func(func_name):
        exec(f"def {func_name}(value): _stub_error()")
        return eval(func_name)

    class _GroupId(AbstractTemplate):
        def generic(self, args, kws):
            assert not kws
            assert len(args) == 1
            elem_type = args[0]

            return signature(elem_type, elem_type)

    this_module = sys.modules[__name__]

    for op in ops:
        func_name = f"group_reduce_{op}"
        short_name = f"reduce_{op}"
        func = get_stub_func(func_name)
        setattr(this_module, func_name, func)

        infer_global(func)(_GroupId)
        registry.register_func(func_name, func)(get_func(func_name))
        setattr(group, short_name, func)


_define_group_funcs()
del _define_group_funcs
