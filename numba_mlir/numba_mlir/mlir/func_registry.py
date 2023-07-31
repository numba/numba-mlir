# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import inspect

_mlir_func_names = {}
_active_funcs_stack = []


def add_func(func, name):
    global _mlir_func_names
    key = id(func)
    assert _mlir_func_names.get(key, name) == name
    _mlir_func_names[key] = name


def get_func_name(func):
    return _mlir_func_names.get(id(func), None)


def push_active_funcs_stack():
    global _active_funcs_stack
    _active_funcs_stack.append({})


def pop_active_funcs_stack():
    global _active_funcs_stack
    assert len(_active_funcs_stack) > 0
    _active_funcs_stack.pop()


def add_active_funcs(name, func, flags):
    global _active_funcs_stack
    assert len(_active_funcs_stack) > 0
    top = _active_funcs_stack[-1]
    params = inspect.signature(func).parameters
    arg_names = list(params.keys())
    empty = inspect.Parameter.empty
    param_values = params.values()
    arg_has_default = list(map(lambda p: p.default is not empty, param_values))
    arg_defaults = list(map(lambda p: p.default, param_values))
    top[name] = (func, flags, arg_names, arg_has_default, arg_defaults)


def find_active_func(name):
    global _active_funcs_stack
    assert len(_active_funcs_stack) > 0
    for elem in reversed(_active_funcs_stack):
        res = elem.get(name)
        if not res is None:
            return res
    return None
