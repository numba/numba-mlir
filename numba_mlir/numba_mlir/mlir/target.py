# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import copy
from functools import singledispatch, cached_property

from numba.core import types, cpu, utils, compiler
from numba.extending import typeof_impl as numba_typeof_impl
from numba.core.typing import Context
from numba.core.registry import CPUTarget
from numba.core.dispatcher import Dispatcher
from numba.core.typing.templates import Registry
from numba.core.typing.typeof import Purpose, _TypeofContext, _termcolor
from numba.core.target_extension import (
    JitDecorator,
    target_registry,
    dispatcher_registry,
    jit_registry,
    CPU,
)

from .compiler import mlir_compiler_pipeline, get_gpu_pipeline


def typeof(val, purpose=Purpose.argument):
    """
    Get the Numba type of a Python value for the given purpose.
    """
    # Note the behaviour for Purpose.argument must match _typeof.c.
    c = _TypeofContext(purpose)
    ty = typeof_impl(val, c)
    if ty is None:
        msg = _termcolor.errmsg(f"Cannot determine Numba type of {type(val)}")
        raise ValueError(msg)
    return ty


@singledispatch
def typeof_impl(val, c):
    """
    Use Numba impl by default
    """
    return numba_typeof_impl(val, c)


registry = Registry()
infer = registry.register
infer_global = registry.register_global
infer_getattr = registry.register_attr


class NumbaMLIR(CPU):
    pass


target_name = "numba-mlir"
target_registry[target_name] = NumbaMLIR


class NumbaMLIRContext(Context):
    def load_additional_registries(self):
        self.install_registry(registry)
        super().load_additional_registries()

    def resolve_argument_type(self, val):
        """
        Return the numba type of a Python value that is being used
        as a function argument.  Integer types will all be considered
        int64, regardless of size.

        ValueError is raised for unsupported types.
        """
        return typeof(val, Purpose.argument)

    def resolve_value_type(self, val):
        """
        Return the numba type of a Python value that is being used
        as a runtime constant.
        ValueError is raised for unsupported types.
        """
        try:
            ty = typeof(val, Purpose.constant)
        except ValueError as e:
            # Make sure the exception doesn't hold a reference to the user
            # value.
            typeof_exc = utils.erase_traceback(e)
        else:
            return ty

        if isinstance(val, types.ExternalFunction):
            return val

        # Try to look up target specific typing information
        ty = self._get_global_type(val)
        if ty is not None:
            return ty

        raise typeof_exc


class NumbaMLIRTarget(CPUTarget):
    @cached_property
    def _toplevel_typing_context(self):
        # Lazily-initialized top-level typing context, for all threads
        return NumbaMLIRContext()


numba_mlir_target = NumbaMLIRTarget(target_name)


class NumbaMLIRDispatcher(Dispatcher):
    targetdescr = numba_mlir_target

    def typeof_pyval(self, val):
        """
        Resolve the Numba type of Python value *val*.
        This is called from numba._dispatcher as a fallback if the native code
        cannot decide the type.
        """
        # Not going through the resolve_argument_type() indirection
        # can save a couple µs.
        try:
            tp = typeof(val, Purpose.argument)
        except ValueError:
            tp = types.pyobject
        else:
            if tp is None:
                tp = types.pyobject
        self._types_active_call.append(tp)
        return tp


dispatcher_registry[target_registry[target_name]] = NumbaMLIRDispatcher


class numba_mlir_jit(JitDecorator):
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def __call__(self, *args):
        assert len(args) < 2
        if args:
            func = args[0]
        else:
            func = self._args[0]
        self.py_func = func
        # wrap in dispatcher
        return self.dispatcher_wrapper()

    def get_dispatcher(self):
        """
        Returns the dispatcher
        """
        return NumbaMLIRDispatcher

    def dispatcher_wrapper(self):
        disp = self.get_dispatcher()
        # Parse self._kwargs here
        options = copy.deepcopy(self._kwargs)
        fp64_truncate = options.get("gpu_fp64_truncate", False)
        assert fp64_truncate in [
            True,
            False,
            "auto",
        ], 'gpu_fp64_truncate supported values are True/False/"auto"'

        use_64bit_index = options.get("gpu_use_64bit_index", True)
        assert use_64bit_index in [
            True,
            False,
        ], "gpu_use_64bit_index supported values are True/False"

        # if options.get("enable_gpu_pipeline", True):
        #     pipeline_class = get_gpu_pipeline(fp64_truncate, use_64bit_index)
        # else:
        #     pipeline_class = mlir_compiler_pipeline

        pipeline_class = compiler.Compiler

        options.pop("gpu_fp64_truncate", None)
        options.pop("gpu_use_64bit_index", None)
        options.pop("enable_gpu_pipeline", None)

        pipeline_class = options.get("pipeline_class", pipeline_class)
        return disp(
            py_func=self.py_func,
            targetoptions=options,
            pipeline_class=pipeline_class,
        )


jit_registry[target_registry[target_name]] = numba_mlir_jit
