# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import functools

import llvmlite.ir
from numba.core import types, cgutils
from numba.core.compiler import Flags, compile_result
from numba.core.compiler_machinery import FunctionPass, register_pass
from numba.core.funcdesc import qualifying_prefix
from numba.np.ufunc.parallel import get_thread_count
import numba.core.types.functions
import numba.parfors.parfor
from contextlib import contextmanager

from .settings import DUMP_IR, OPT_LEVEL, DUMP_DIAGNOSTICS
from . import func_registry
from .. import mlir_compiler
from .compiler_context import global_compiler_context


_print_before = []
_print_after = []
_print_buffer = ""


def write_print_buffer(text):
    global _print_buffer
    _print_buffer += text


def get_print_buffer():
    global _print_buffer
    if len(_print_buffer) == 0:
        raise ValueError("Pass print buffer is empty")

    return _print_buffer


def is_print_buffer_empty():
    global _print_buffer
    return len(_print_buffer) == 0


@contextmanager
def print_pass_ir(print_before, print_after):
    global _print_before
    global _print_after
    global _print_buffer
    old_before = _print_before
    old_after = _print_after
    old_buffer = _print_buffer
    _print_before = print_before
    _print_after = print_after
    _print_buffer = ""
    try:
        yield (print_before, print_after)
    finally:
        _print_before = old_before
        _print_after = old_after
        _print_buffer = old_buffer


_mlir_last_compiled_func = None
_mlir_active_module = None


def _create_flags(fp64_truncate, use_64bit_index):
    flags = Flags()
    flags.nrt = True
    setattr(flags, "fp64_truncate", fp64_truncate)
    setattr(flags, "use_64bit_index", use_64bit_index)
    return flags


class MlirBackendBase(FunctionPass):
    def __init__(self, push_func_stack):
        self._push_func_stack = push_func_stack
        self._get_func_name = func_registry.get_func_name
        self._fp64_truncate = False
        self._use_64bit_index = True
        FunctionPass.__init__(self)

    def run_pass(self, state):
        if self._push_func_stack:
            func_registry.push_active_funcs_stack()
            try:
                res = self.run_pass_impl(state)
            finally:
                func_registry.pop_active_funcs_stack()
            return res
        else:
            return self.run_pass_impl(state)

    def _resolve_func_name(self, obj):
        name, func, flags = self._resolve_func_impl(obj)
        if not (name is None or func is None):
            func_registry.add_active_funcs(name, func, flags)
        return name

    def _resolve_func_impl(self, obj):
        if isinstance(obj, types.Function):
            func = obj.typing_key
            return (
                self._get_func_name(func),
                None,
                _create_flags(self._fp64_truncate, self._use_64bit_index),
            )
        if isinstance(obj, types.BoundFunction):
            return (
                str(obj.typing_key),
                None,
                _create_flags(self._fp64_truncate, self._use_64bit_index),
            )
        if isinstance(obj, numba.core.types.functions.Dispatcher):
            flags = _create_flags(self._fp64_truncate, self._use_64bit_index)
            func = obj.dispatcher.py_func
            inline_type = obj.dispatcher.targetoptions.get("inline", None)
            if inline_type is not None:
                flags.inline._inline = inline_type

            parallel_type = obj.dispatcher.targetoptions.get("parallel", None)
            if parallel_type is not None:
                flags.auto_parallel = parallel_type

            fastmath_type = obj.dispatcher.targetoptions.get("fastmath", None)
            if fastmath_type is not None:
                flags.fastmath = fastmath_type

            return (func.__module__ + "." + func.__qualname__, func, flags)
        if isinstance(obj, types.NumberClass):
            return (
                "$number." + str(obj.instance_type),
                None,
                _create_flags(self._fp64_truncate, self._use_64bit_index),
            )
        return (None, None, None)

    def _get_func_context(self, state):
        mangler = state.targetctx.mangler
        mangler = default_mangler if mangler is None else mangler
        unique_name = state.func_ir.func_id.unique_name
        modname = state.func_ir.func_id.func.__module__
        qualprefix = qualifying_prefix(modname, unique_name)
        abi_tags = [state.flags.get_mangle_string()]
        fn_name = mangler(qualprefix, state.args, abi_tags=abi_tags)

        ctx = {}
        ctx["compiler_settings"] = {
            "verify": True,
            "pass_statistics": False,
            "pass_timings": False,
            "ir_printing": DUMP_IR,
            "diag_printing": DUMP_DIAGNOSTICS,
            "print_before": _print_before,
            "print_after": _print_after,
            "print_callback": write_print_buffer,
        }
        ctx["typemap"] = lambda op: state.typemap[op.name]
        ctx["fnargs"] = lambda: state.args
        ctx["restype"] = lambda: state.return_type
        ctx["fnname"] = lambda: fn_name
        ctx["resolve_func"] = self._resolve_func_name
        ctx["globals"] = lambda: state.func_id.func.__globals__

        func_attrs = {}
        if state.targetctx.fastmath:
            func_attrs["numba.fastmath"] = None

        if state.flags.inline.is_always_inline:
            func_attrs["numba.force_inline"] = None

        if state.flags.auto_parallel.enabled:
            func_attrs["numba.max_concurrency"] = get_thread_count()

        func_attrs["numba.opt_level"] = OPT_LEVEL

        if self._fp64_truncate != "auto":
            func_attrs["gpu_runtime.fp64_truncate"] = self._fp64_truncate

        func_attrs["gpu_runtime.use_64bit_index"] = self._use_64bit_index

        ctx["func_attrs"] = func_attrs
        return ctx


@register_pass(mutates_CFG=True, analysis_only=False)
class MlirDumpPlier(MlirBackendBase):
    _name = "mlir_dump_plier"

    def __init__(self):
        MlirBackendBase.__init__(self, push_func_stack=True)

    def run_pass(self, state):
        module = mlir_compiler.create_module()
        ctx = self._get_func_context(state)
        mlir_compiler.lower_function(ctx, module, state.func_ir)
        print(mlir_compiler.module_str(module))
        return True


def get_mlir_func():
    global _mlir_last_compiled_func
    return _mlir_last_compiled_func


@register_pass(mutates_CFG=True, analysis_only=False)
class MlirBackend(MlirBackendBase):
    _name = "mlir_backend"

    def __init__(self):
        MlirBackendBase.__init__(self, push_func_stack=True)
        self.enable_gpu_pipeline = False

    def run_pass_impl(self, state):
        global _mlir_active_module
        old_module = _mlir_active_module

        try:
            mod_settings = {"enable_gpu_pipeline": self.enable_gpu_pipeline}
            module = mlir_compiler.create_module(mod_settings)
            _mlir_active_module = module
            global _mlir_last_compiled_func
            ctx = self._get_func_context(state)
            _mlir_last_compiled_func = mlir_compiler.lower_function(
                ctx, module, state.func_ir
            )

            # TODO: properly handle returned module ownership
            compiled_mod = mlir_compiler.compile_module(
                global_compiler_context, ctx, module
            )
            func_name = ctx["fnname"]()
            func_ptr = mlir_compiler.get_function_pointer(
                global_compiler_context, compiled_mod, func_name
            )
        finally:
            _mlir_active_module = old_module
        state.metadata["mlir_func_ptr"] = func_ptr
        state.metadata["mlir_func_name"] = func_name
        return True


@functools.lru_cache
def get_gpu_backend(fp64_trunc, use_64bit_index):
    class MlirBackendGPU(MlirBackend):
        def __init__(self):
            MlirBackend.__init__(self)
            self.enable_gpu_pipeline = True
            self._fp64_truncate = fp64_trunc
            self._use_64bit_index = use_64bit_index

    return register_pass(mutates_CFG=True, analysis_only=False)(MlirBackendGPU)


@register_pass(mutates_CFG=True, analysis_only=False)
class MlirBackendInner(MlirBackendBase):
    _name = "mlir_backend_inner"

    def __init__(self):
        MlirBackendBase.__init__(self, push_func_stack=False)

    def run_pass_impl(self, state):
        global _mlir_active_module
        module = _mlir_active_module
        assert not module is None
        global _mlir_last_compiled_func
        ctx = self._get_func_context(state)
        _mlir_last_compiled_func = mlir_compiler.lower_function(
            ctx, module, state.func_ir
        )
        state.cr = compile_result()
        return True


@functools.lru_cache
def get_inner_backend(fp64_trunc, use_64bit_index):
    class MlirBackendInner(MlirBackendBase):
        _name = "mlir_backend_inner"

        def __init__(self):
            MlirBackendBase.__init__(self, push_func_stack=False)
            self._fp64_truncate = fp64_trunc
            self._64bit_index = use_64bit_index

        def run_pass_impl(self, state):
            global _mlir_active_module
            module = _mlir_active_module
            assert not module is None
            global _mlir_last_compiled_func
            ctx = self._get_func_context(state)
            _mlir_last_compiled_func = mlir_compiler.lower_function(
                ctx, module, state.func_ir
            )
            state.cr = compile_result()
            return True

    return register_pass(mutates_CFG=True, analysis_only=False)(MlirBackendInner)


@register_pass(mutates_CFG=True, analysis_only=False)
class MlirReplaceParfors(MlirBackendBase):
    _name = "mlir_replace_parfors"

    def __init__(self):
        MlirBackendBase.__init__(self, push_func_stack=False)
        try:
            from numba_dpex.core.types import USMNdArray
            from .dpctl_interop import _get_device_caps
            import dpctl
            self._usmarray_type = USMNdArray
            self._get_device_caps = _get_device_caps
            self._device_ctor = dpctl.SyclDevice
        except ImportError:
            self._usmarray_type = None;

    def run_pass(self, state):
        print("-=-=-=-=-=- MlirReplaceParfors -=-=-=-=-=-")
        ir = state.func_ir
        ir.dump()
        module = None
        parfor_funcs = {}
        for _, block in ir.blocks.items():
            for inst in block.body:
                if not isinstance(inst, numba.parfors.parfor.Parfor):
                    continue

                inst.dump()
                if module is None:
                    mod_settings = {"enable_gpu_pipeline": True}
                    module = mlir_compiler.create_module(mod_settings)

                typemap = state.typemap
                fn_name = f"parfor_impl{inst.id}"
                arg_types = self._get_parfor_args_types(typemap, inst)
                res_type = self._get_parfor_return_type(typemap, inst)
                device_caps = self._get_parfor_device_caps(arg_types)

                ctx = self._get_func_context(state)
                ctx["fnname"] = lambda: fn_name
                ctx["fnargs"] = lambda: arg_types
                ctx["restype"] = lambda: res_type

                ctx["device_caps"] = device_caps

                mlir_compiler.lower_parfor(ctx, module, inst)
                parfor_funcs[inst] = fn_name

        if not module:
            return False

        compiled_mod = mlir_compiler.compile_module(
            global_compiler_context, ctx, module
        )

        for inst, func_name in parfor_funcs.items():
            func_ptr = mlir_compiler.get_function_pointer(
                global_compiler_context, compiled_mod, func_name
            )
            inst.lowerer = functools.partial(self._lower_parfor, func_ptr)

        return True

    def _enumerate_parfor_args(self, parfor, func):
        ret = []
        for param in parfor.params:
            ret += func(param)

        for loop in parfor.loop_nests:
            for v in (loop.start, loop.stop, loop.step):
                if isinstance(v, int):
                    continue

                ret += func(v.name)

        return ret

    def _get_parfor_args_types(self, typemap, parfor):
        return self._enumerate_parfor_args(parfor, lambda v: [typemap[v]])

    def _get_parfor_device_caps(self, types):
        if self._usmarray_type is None:
            return None

        for t in types:
            if isinstance(t, self._usmarray_type):
                return self._get_device_caps(self._device_ctor(t.device))

        return None

    def _get_parfor_return_type(self, typemap, parfor):
        ret = []
        for param in parfor.redvars:
            ret.append(typemap[param])

        count = len(ret)
        if count == 0:
            return types.none

        if count == 1:
            return ret[0]

        return types.Tuple(ret)

    def _lower_parfor(self, func_ptr, lowerer, parfor):
        print("-=-=-=-=-=-=- lowerer")
        print(lowerer)

        context = lowerer.context
        builder = lowerer.builder
        typemap = lowerer.fndesc.typemap

        res_type = self._get_parfor_return_type(typemap, parfor)
        res_type = context.get_value_type(res_type)

        nullptr = context.get_constant_null(types.voidptr)

        args = []

        # First arg is pointer to return value(s) storage.
        res_storage = cgutils.alloca_once(builder, res_type)
        args.append(res_storage)

        # Second arg is exception info - exceptions is not implemented yet.
        args.append(nullptr)

        def get_arg(v):
            return self._repack_arg(builder, typemap[v], lowerer.loadvar(v))

        args += self._enumerate_parfor_args(parfor, get_arg)

        fnty = llvmlite.ir.FunctionType(llvmlite.ir.IntType(32), [a.type for a in args])
        print(fnty)
        print(func_ptr)

        fnptr = context.get_constant(types.uintp, func_ptr)
        fnptr = builder.inttoptr(fnptr, llvmlite.ir.PointerType(fnty))
        print(fnptr)
        status = builder.call(fnptr, args)
        print(status)
        # Func returns exception status, but exceptions are not implemented yet.

        # Unpack reduction values
        red_vals = []
        num_reds = len(parfor.redvars)
        if num_reds == 0:
            # nothing
            pass
        elif num_reds == 1:
            red_vals.append(builder.load(res_storage))
        else:
            struct = builder.load(res_storage)
            for i in range(num_reds):
                red_vals.append(builder.extract_value(struct, i))

        for red_val, red_var in zip(red_vals, parfor.redvars):
            lowerer.storevar(red_val, name=red_var)

    def _repack_arg(self, builder, orig_type, arg):
        ret = []
        typ = arg.type
        if isinstance(typ, llvmlite.ir.BaseStructType):
            usmarray_type = self._usmarray_type
            is_usm_array = usmarray_type and isinstance(orig_type, usmarray_type)
            print('asdsadasdsasdsa', is_usm_array)
            # USM array data model mostly follows numpy model except additional
            # sycl queue pointer. Queue can be extracted from meminfo, so just
            # skip it.
            # TODO: Actually parse data models instead of hardcoding index?
            for i in range(len(typ)):
                if is_usm_array and i == 5:
                    continue

                ret.append(builder.extract_value(arg, i))
        else:
            ret.append(arg)

        return ret
