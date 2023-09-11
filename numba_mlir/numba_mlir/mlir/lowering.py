# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Define lowering and related passes.
"""

from .passes import MlirDumpPlier, MlirBackend
from .settings import USE_MLIR

from numba.core.compiler_machinery import register_pass

from numba.core import types
from numba.core.lowering import Lower as orig_Lower
from numba.core.typed_passes import NativeLowering as orig_NativeLowering

# looks like that we don't need it but it is inherited from BaseLower too
# from numba.core.pylowering import PyLower as orig_PyLower

from .runtime import *
from .math_runtime import *
from .numba_runtime import *
from .gpu_runtime import *

import llvmlite.ir as ir
import llvmlite.binding as llvm


class mlir_lower(orig_Lower):
    def lower(self):
        if USE_MLIR:
            self.emit_environment_object()
            self.genlower = None
            self.lower_normal_function(self.fndesc)
            self.context.post_lowering(self.module, self.library)

            # Skip check that all numba symbols defined
            setattr(self.library, "_verify_declare_only_symbols", lambda: None)

            self.library.add_ir_module(self.module)
        else:
            super().lower(self)

    def lower_normal_function(self, fndesc):
        if USE_MLIR:
            self.setup_function(fndesc)
            builder = self.builder
            context = self.context

            fnty = self.call_conv.get_function_type(fndesc.restype, fndesc.argtypes)
            func_ptr = self.metadata.pop("mlir_func_ptr")
            func_ptr = context.get_constant(types.uintp, func_ptr)
            func_ptr = builder.inttoptr(func_ptr, ir.PointerType(fnty))

            ret = builder.call(func_ptr, self.function.args)
            builder.ret(ret)
        else:
            super().lower_normal_function(self, desc)


@register_pass(mutates_CFG=True, analysis_only=False)
class mlir_NativeLowering(orig_NativeLowering):
    @property
    def lowering_class(self):
        return mlir_lower


@register_pass(mutates_CFG=True, analysis_only=False)
class dummy_NativeLowering(mlir_NativeLowering):
    def run_pass(self, state):
        state.metadata["mlir_func_ptr"] = 1
        return super().run_pass(self, state)
