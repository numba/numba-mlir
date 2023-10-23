# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ..linalg_builder import (
    FuncRegistry,
    is_int,
    is_float,
    is_complex,
    broadcast_type,
    literal,
)
from ..func_registry import add_func
from . import helper_funcs

import math
from numba.cpython.builtins import get_type_min_value, get_type_max_value
from numba.parfors.array_analysis import wrap_index, assert_equiv
from numba.parfors.parfor import (
    max_checker,
    min_checker,
    argmax_checker,
    argmin_checker,
)

add_func(slice, "slice")


registry = FuncRegistry()


def register_func(name, orig_func=None):
    global registry
    return registry.register_func(name, orig_func)


@register_func("range", range)
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


@register_func("bool", bool)
def bool_cast_impl(builder, arg):
    return builder.cast(arg, builder.bool)


@register_func("int", int)
def int_cast_impl(builder, arg):
    return builder.cast(arg, builder.int64)


@register_func("float", float)
def float_cast_impl(builder, arg):
    return builder.cast(arg, builder.float64)


def _gen_casts():
    types = [
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float32",
        "float64",
        "complex64",
        "complex128",
    ]

    def gen_cast_func(type_name):
        def func(builder, arg):
            t = getattr(builder, type_name)
            return builder.cast(arg, t)

        return func

    for type_name in types:
        func_name = f"$number.{type_name}"
        register_func(func_name)(gen_cast_func(type_name))


_gen_casts()
del _gen_casts


@register_func("len", len)
def len_impl(builder, arg):
    try:
        l = len(arg)
    except:
        return None

    return builder.cast(l, builder.int64)


def _get_type(builder, v):
    if isinstance(v, float):
        return builder.float64
    elif isinstance(v, int):
        return builder.int64
    return v.type


@register_func("min", min)
def min_impl(builder, *args):
    if len(args) > 2:
        rhs = min_impl(builder, *args[1:])
    else:
        rhs = args[1]

    lhs = args[0]
    res_type = broadcast_type(
        builder, (_get_type(builder, lhs), _get_type(builder, rhs))
    )
    lhs = builder.cast(lhs, res_type)
    rhs = builder.cast(rhs, res_type)
    cond = lhs < rhs
    return builder.select(cond, lhs, rhs)


@register_func("max", max)
def max_impl(builder, *args):
    if len(args) > 2:
        rhs = max_impl(builder, *args[1:])
    else:
        rhs = args[1]

    lhs = args[0]
    res_type = broadcast_type(
        builder, (_get_type(builder, lhs), _get_type(builder, rhs))
    )
    lhs = builder.cast(lhs, res_type)
    rhs = builder.cast(rhs, res_type)
    cond = lhs > rhs
    return builder.select(cond, lhs, rhs)


def _gen_math_funcs():
    def get_func(name, N):
        def func(builder, *args):
            if len(args) != N:
                return None

            t = args[0].type
            if not is_int(t, builder) and not is_float(t, builder):
                return None

            for a in args[1:]:
                if a.type != t:
                    return None

            fname = name
            if t == builder.float32:
                fname = "f" + fname
            elif t != builder.float64:
                t = builder.float64
                args = tuple(builder.cast(arg, builder.float64) for arg in args)

            res = builder.cast(0, t)
            return builder.external_call(fname, args, res, decorate=False)

        return func

    math_funcs = [
        ("floor", 1),
        ("log", 1),
        ("sqrt", 1),
        ("exp", 1),
        ("erf", 1),
        ("sin", 1),
        ("cos", 1),
        ("tanh", 1),
        ("atan2", 2),
    ]

    for func, N in math_funcs:
        fname = "math." + func
        py_func = eval(fname)
        register_func(fname, py_func)(get_func(func, N))


_gen_math_funcs()
del _gen_math_funcs


def _get_math_func_name(builder, name, t, append_f=False):
    if is_float(t, builder):
        fname = name
        if append_f:
            fname = "f" + fname
        if t == builder.float32:
            fname = fname + "f"
        return fname

    if is_complex(t, builder):
        fname = "c" + name
        if t == builder.complex64:
            fname = fname + "f"
        return fname

    assert False, "Unsupported type"


def _get_complex_elem_type(builder, t):
    if t == builder.complex64:
        return builder.float32
    if t == builder.complex128:
        return builder.float64

    return t


@register_func("abs", abs)
def abs_impl(builder, arg):
    t = arg.type
    if is_int(t, builder):
        c = arg < 0
        return builder.select(c, -arg, arg)

    fname = _get_math_func_name(builder, "abs", t, append_f=True)
    res_type = _get_complex_elem_type(builder, t)

    res = builder.cast(0, res_type)
    return builder.external_call(fname, arg, res, decorate=False)


def _build_math_func(builder, arg, name):
    t = arg.type
    if is_int(t, builder):
        t = builder.float64
        arg = builder.cast(arg, t)

    fname = _get_math_func_name(builder, name, t)

    res = builder.cast(0, t)
    return builder.external_call(fname, arg, res, decorate=False)


@register_func("mlir.helper_funcs.exp", helper_funcs.exp)
def exp_impl(builder, arg):
    return _build_math_func(builder, arg, "exp")


@register_func("mlir.helper_funcs.sqrt", helper_funcs.sqrt)
def sqrt_impl(builder, arg):
    return _build_math_func(builder, arg, "sqrt")


@register_func("mlir.helper_funcs.sin", helper_funcs.sin)
def exp_impl(builder, arg):
    return _build_math_func(builder, arg, "sin")


@register_func("mlir.helper_funcs.cos", helper_funcs.cos)
def sqrt_impl(builder, arg):
    return _build_math_func(builder, arg, "cos")


def _get_min_max_values(builder, dtype):
    values = [
        (builder.int8, -128, 127),
        (builder.uint8, 0, 255),
        (builder.int16, -32768, 32767),
        (builder.uint16, 0, 65535),
        (builder.int32, -2147483648, 2147483647),
        (builder.uint32, 0, 4294967295),
        (builder.int64, -9223372036854775808, 9223372036854775807),
        (builder.uint64, 0, 18446744073709551615),
        (builder.float32, -math.inf, math.inf),
        (builder.float64, -math.inf, math.inf),
    ]
    for t, min_val, max_val in values:
        if t == dtype:
            return (min_val, max_val)

    return None


@register_func("builtin.get_type_min_value", get_type_min_value)
def get_type_min_value_impl(builder, dtype):
    val = _get_min_max_values(builder, dtype)
    if val is None:
        return

    return val[0]


@register_func("builtin.get_type_max_value", get_type_max_value)
def get_type_max_value_impl(builder, dtype):
    val = _get_min_max_values(builder, dtype)
    if val is None:
        return

    return val[1]


@register_func("parfor.wrap_index", wrap_index)
def wrap_index_impl(builder, idx, size):
    neg = idx < 0
    idx = builder.select(neg, size - idx, idx)
    undeflow = idx < 0
    overflow = idx > size
    idx = builder.select(undeflow, 0, idx)
    idx = builder.select(overflow, size, idx)
    return idx


@register_func("parfor.assert_equiv", assert_equiv)
def wrap_index_impl(builder, *val):
    # TODO: implement
    return 0


@register_func("parfor.max_checker", max_checker)
@register_func("parfor.min_checker", min_checker)
@register_func("parfor.argmax_checker", argmax_checker)
@register_func("parfor.argmin_checker", argmin_checker)
def parfor_checker(buidler, size):
    # TODO: implement
    return 0
