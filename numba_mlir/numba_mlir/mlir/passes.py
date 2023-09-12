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
import numba.core.ir as ir
from numba.core.ir_utils import mk_unique_var
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
    setattr(flags, "gpu_fp64_truncate", fp64_truncate)
    setattr(flags, "gpu_use_64bit_index", use_64bit_index)
    return flags


class MlirBackendBase(FunctionPass):
    def __init__(self, push_func_stack):
        self._push_func_stack = push_func_stack
        self._get_func_name = func_registry.get_func_name
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

    def _resolve_func_name(self, state, obj):
        name, func, flags = self._resolve_func_impl(state, obj)
        if not (name is None or func is None):
            func_registry.add_active_funcs(name, func, flags)
        return name

    def _resolve_func_impl(self, state, obj):
        fp64_truncate = state.flags.gpu_fp64_truncate
        use_64bit_index = state.flags.gpu_use_64bit_index
        if isinstance(obj, types.Function):
            func = obj.typing_key
            return (
                self._get_func_name(func),
                None,
                _create_flags(fp64_truncate, use_64bit_index),
            )
        if isinstance(obj, types.BoundFunction):
            return (
                str(obj.typing_key),
                None,
                _create_flags(fp64_truncate, use_64bit_index),
            )
        if isinstance(obj, numba.core.types.functions.Dispatcher):
            flags = _create_flags(fp64_truncate, use_64bit_index)
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
                _create_flags(fp64_truncate, use_64bit_index),
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
        ctx["resolve_func"] = lambda obj: self._resolve_func_name(state, obj)
        ctx["globals"] = lambda: state.func_id.func.__globals__

        func_attrs = {}
        if state.targetctx.fastmath:
            func_attrs["numba.fastmath"] = None

        if state.flags.inline.is_always_inline:
            func_attrs["numba.force_inline"] = None

        if state.flags.auto_parallel.enabled:
            func_attrs["numba.max_concurrency"] = get_thread_count()

        func_attrs["numba.opt_level"] = OPT_LEVEL

        if state.flags.gpu_fp64_truncate != "auto":
            func_attrs["gpu_runtime.fp64_truncate"] = state.flags.gpu_fp64_truncate

        func_attrs["gpu_runtime.use_64bit_index"] = state.flags.gpu_use_64bit_index

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

    def run_pass_impl(self, state):
        global _mlir_active_module
        old_module = _mlir_active_module

        try:
            mod_settings = {"enable_gpu_pipeline": state.flags.enable_gpu_pipeline}
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
def get_gpu_backend():
    class MlirBackendGPU(MlirBackend):
        def __init__(self):
            MlirBackend.__init__(self)

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
def get_inner_backend():
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
            self._usmarray_type = None

    def run_pass(self, state):
        global _mlir_active_module
        old_module = _mlir_active_module
        func_registry.push_active_funcs_stack()
        try:
            module, parfor_funcs, ctx = self._gen_module(state)

            if not module:
                return False

            _mlir_active_module = module

            compiled_mod = mlir_compiler.compile_module(
                global_compiler_context, ctx, module
            )
        finally:
            func_registry.pop_active_funcs_stack()
            _mlir_active_module = old_module

        for inst, func_name in parfor_funcs.items():
            func_ptr = mlir_compiler.get_function_pointer(
                global_compiler_context, compiled_mod, func_name
            )
            inst.lowerer = functools.partial(self._lower_parfor, func_ptr)

        return True

    def _gen_module(self, state):
        ir = state.func_ir
        module = None
        parfor_funcs = {}
        ctx = None
        for _, block in ir.blocks.items():
            for inst in block.body:
                if not isinstance(inst, numba.parfors.parfor.Parfor):
                    continue

                typemap = state.typemap
                self._update_parfor_redvars(inst, state)
                self._reconstruct_parfor_ssa(inst, typemap)

                if module is None:
                    mod_settings = {"enable_gpu_pipeline": True}
                    module = mlir_compiler.create_module(mod_settings)

                fn_name = f"parfor_impl{inst.id}"
                arg_types = self._get_parfor_args_types(typemap, inst)
                res_type = self._get_parfor_return_type(typemap, inst)
                device_caps = self._get_parfor_device_caps(arg_types)

                ctx = self._get_func_context(state)
                ctx["fnname"] = lambda: fn_name
                ctx["fnargs"] = lambda: arg_types
                ctx["restype"] = lambda: res_type

                ctx["device_caps"] = device_caps

                output_arrays = numba.parfors.parfor.get_parfor_outputs(
                    inst, inst.params
                )
                ctx["parfor_params"] = self._get_parfor_params(inst)
                ctx["parfor_output_arrays"] = output_arrays

                mlir_compiler.lower_parfor(ctx, module, inst)
                parfor_funcs[inst] = fn_name

        return module, parfor_funcs, ctx

    def _update_parfor_redvars(self, parfor, state):
        parfor.redvars, parfor.reddict = numba.parfors.parfor.get_parfor_reductions(
            state.func_ir, parfor, parfor.params, state.calltypes
        )

        for block in parfor.loop_body.values():
            for inst in block.body:
                if isinstance(inst, numba.parfors.parfor.Parfor):
                    self._update_parfor_redvars(inst, state)

    def _reconstruct_parfor_ssa(self, parfor, typemap):
        loop_body = parfor.loop_body

        # Add dummy return
        last_label = max(loop_body.keys())
        scope = loop_body[last_label].scope
        const = ir.Var(scope, mk_unique_var("$const"), ir.Loc("parfors_dummy", -1))
        loop_body[last_label].body.append(ir.Return(const, ir.Loc("parfors_dummy", -1)))

        # Reconstruct loop body SSA
        loop_body = numba.core.ssa._run_ssa(loop_body)

        # remove dummy return
        loop_body[last_label].body.pop()
        parfor.loop_body = loop_body

        for block in loop_body.values():
            for inst in block.body:
                if (
                    isinstance(inst, ir.Assign)
                    and isinstance(inst.value, ir.Expr)
                    and inst.value.op == "phi"
                ):
                    for inc in inst.value.incoming_values:
                        try:
                            t = typemap[inc.name]
                        except KeyError:
                            continue
                        typemap[inst.target.name] = t
                        break

                    for inc in inst.value.incoming_values:
                        if inc.name not in typemap:
                            typemap[inc.name] = t
                elif isinstance(inst, numba.parfors.parfor.Parfor):
                    self._reconstruct_parfor_ssa(inst, typemap)

    def _get_parfor_params(self, parfor):
        params = []
        usedefs = numba.core.analysis.compute_use_defs({0: parfor.init_block})

        init_block_uses = usedefs.usemap[0]
        init_block_defs = usedefs.defmap[0]

        params += sorted(list(init_block_uses))

        for param in parfor.params:
            if param in init_block_defs:
                continue

            params.append(param)

        return params

    def _enumerate_parfor_args(self, parfor, func):
        ret = []

        for param in self._get_parfor_params(parfor):
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
        output_arrays = numba.parfors.parfor.get_parfor_outputs(parfor, parfor.params)

        for out in output_arrays:
            ret.append(typemap[out])

        for param in parfor.redvars:
            ret.append(typemap[param])

        count = len(ret)
        if count == 0:
            return types.none

        if count == 1:
            return ret[0]

        return types.Tuple(ret)

    def _lower_parfor(self, func_ptr, lowerer, parfor):
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

        # Second arg is exception info - exceptions are not implemented yet.
        args.append(nullptr)

        def get_arg(v):
            return self._repack_arg(builder, typemap[v], lowerer.loadvar(v))

        args += self._enumerate_parfor_args(parfor, get_arg)

        fnty = llvmlite.ir.FunctionType(llvmlite.ir.IntType(32), [a.type for a in args])

        fnptr = context.get_constant(types.uintp, func_ptr)
        fnptr = builder.inttoptr(fnptr, llvmlite.ir.PointerType(fnty))

        status = builder.call(fnptr, args)
        # Func returns exception status, but exceptions are not implemented yet.

        # Unpack returned values
        ret_vals = []

        output_arrays = numba.parfors.parfor.get_parfor_outputs(parfor, parfor.params)

        num_rets = len(output_arrays) + len(parfor.redvars)
        if num_rets == 0:
            # nothing
            pass
        elif num_rets == 1:
            ret_vals.append(builder.load(res_storage))
        else:
            struct = builder.load(res_storage)
            for i in range(num_rets):
                ret_vals.append(builder.extract_value(struct, i))

        for ret_val, ret_var in zip(ret_vals, output_arrays + parfor.redvars):
            lowerer.storevar(ret_val, name=ret_var)

    def _repack_arg(self, builder, orig_type, arg):
        ret = []
        typ = arg.type
        if isinstance(typ, llvmlite.ir.BaseStructType):
            usmarray_type = self._usmarray_type
            is_usm_array = usmarray_type and isinstance(orig_type, usmarray_type)
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
