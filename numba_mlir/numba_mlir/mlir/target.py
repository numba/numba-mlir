# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import threading
from enum import Enum
from functools import singledispatch, cached_property
from contextlib import contextmanager

from numba.core import types, cpu, utils, compiler, options
from numba.extending import typeof_impl as numba_typeof_impl
from numba.core.typing import Context
from numba.core.registry import CPUTarget
from numba.core.imputils import Registry as LowerRegistry
from numba.core.dispatcher import Dispatcher, _FunctionCompiler
from numba.core.typing.templates import Registry as TypingRegistry
from numba.core.typing.typeof import Purpose, _TypeofContext, _termcolor
from numba.core.target_extension import (
    JitDecorator,
    target_registry,
    dispatcher_registry,
    jit_registry,
    CPU,
)


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


@typeof_impl.register(tuple)
def _typeof_tuple(val, c):
    tys = [typeof_impl(v, c) for v in val]
    if any(ty is None for ty in tys):
        return
    return types.BaseTuple.from_types(tys, type(val))


target_name = "numba-mlir"

typing_registry = TypingRegistry()
infer = typing_registry.register
infer_global = typing_registry.register_global
infer_getattr = typing_registry.register_attr


lower_registry = LowerRegistry(target_name)


class NumbaMLIR(CPU):
    pass


target_registry[target_name] = NumbaMLIR


class NumbaMLIRContext(cpu.CPUContext):
    def __init__(self, typingctx, target=target_name):
        super().__init__(typingctx, target)

    def load_additional_registries(self):
        self.install_registry(lower_registry)
        super().load_additional_registries()


class NumbaMLIRTypingContext(Context):
    def load_additional_registries(self):
        self.install_registry(typing_registry)
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


_option_mapping = options._mapping


class F64Truncate(Enum):
    Always = True
    Never = False
    Auto = "auto"


def _map_f64truncate(val):
    if val is True:
        return F64Truncate.Always
    elif val is False:
        return F64Truncate.Never
    elif val == "auto":
        return F64Truncate.Auto
    else:
        raise ValueError(f"Invalid f64 truncate value: {val}")


def _get_host_vec_length():
    from .settings import DISABLE_VECTORIZE

    if DISABLE_VECTORIZE:
        return 0

    from ..mlir_compiler import get_vector_length

    return get_vector_length()


_def_vector_len = _get_host_vec_length()


def _map_vectorize(val):
    if isinstance(val, int):
        return val

    if val is True:
        _def_vector_len

    if val is False:
        return 0

    raise ValueError(f"Invalid mlir_vectorize value: {val}")


def _set_option(flags, name, options, default, mapping=lambda a: a):
    value = mapping(options.get(name, default))
    setattr(flags, name, value)


class NumbaMLIRTargetOptions(cpu.CPUTargetOptions):
    gpu_fp64_truncate = _option_mapping("gpu_fp64_truncate", _map_f64truncate)
    gpu_use_64bit_index = _option_mapping("gpu_use_64bit_index")
    enable_gpu_pipeline = _option_mapping("enable_gpu_pipeline")
    mlir_force_inline = _option_mapping("mlir_force_inline")
    mlir_vectorize = _option_mapping("mlir_vectorize", _map_vectorize)

    def finalize(self, flags, options):
        super().finalize(flags, options)
        _set_option(flags, "gpu_fp64_truncate", options, False)
        _set_option(flags, "gpu_use_64bit_index", options, True)
        _set_option(flags, "enable_gpu_pipeline", options, True)
        _set_option(flags, "mlir_force_inline", options, False)
        _set_option(flags, "mlir_vectorize", options, _def_vector_len)
        assert flags.gpu_fp64_truncate in [
            True,
            False,
            "auto",
        ], 'gpu_fp64_truncate supported values are True/False/"auto"'
        assert flags.gpu_use_64bit_index in [
            True,
            False,
        ], "gpu_use_64bit_index supported values are True/False"
        assert flags.enable_gpu_pipeline in [
            True,
            False,
        ], "enable_gpu_pipeline supported values are True/False"


class NumbaMLIRTarget(CPUTarget):
    options = NumbaMLIRTargetOptions

    @cached_property
    def _toplevel_target_context(self):
        # Lazily-initialized top-level target context, for all threads
        return NumbaMLIRContext(self.typing_context, self._target_name)

    @cached_property
    def _toplevel_typing_context(self):
        # Lazily-initialized top-level typing context, for all threads
        return NumbaMLIRTypingContext()


numba_mlir_target = NumbaMLIRTarget(target_name)


tls_state = threading.local()
tls_state.compiler_nest = 0


@contextmanager
def compile_scope():
    tls_state.compiler_nest += 1
    try:
        yield tls_state.compiler_nest
    finally:
        tls_state.compiler_nest -= 1


def is_nested_compile():
    return tls_state.compiler_nest > 0


class NumbaMLIRDispatcher(Dispatcher):
    targetdescr = numba_mlir_target

    def __init__(
        self,
        py_func,
        locals={},
        targetoptions={},
        pipeline_class=compiler.Compiler,
    ):
        super().__init__(py_func, locals, targetoptions, pipeline_class)

        # Import locally to avoid circular module dependency
        from .compiler import dummy_compiler_pipeline

        compiler_class = _FunctionCompiler
        self._dummy_compiler = compiler_class(
            py_func, self.targetdescr, targetoptions, locals, dummy_compiler_pipeline
        )

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

    def compile(self, *args, **kwargs):
        if is_nested_compile():
            return self._dummy_compile(*args, **kwargs)

        with compile_scope() as s:
            return super().compile(*args, **kwargs)

    def _dummy_compile(self, *args, **kwargs):
        old_compiler = self._compiler
        self._compiler = self._dummy_compiler
        try:
            return super().compile(*args, **kwargs)
        finally:
            self._compiler = old_compiler


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
        # Import locally to avoid circular module dependency
        from .compiler import mlir_compiler_pipeline, dummy_compiler_pipeline

        disp = self.get_dispatcher()
        # Parse self._kwargs here
        options = self._kwargs
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

        # pipeline_class = mlir_compiler_pipeline
        # pipeline_class = compiler.Compiler
        pipeline_class = dummy_compiler_pipeline

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
