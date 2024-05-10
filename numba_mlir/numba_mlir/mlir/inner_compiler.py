# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from numba.core.untyped_passes import ReconstructSSA
from numba.core.typed_passes import NopythonTypeInference
from numba.core.compiler import (
    compile_extra,
    CompilerBase,
    DefaultPassBuilder,
    Flags,
)
from numba.core.compiler_machinery import PassManager
from numba.core import typing, cpu

from numba_mlir.mlir.passes import MlirBackendInner, get_mlir_func

from .target import numba_mlir_target


class MlirTempCompiler(CompilerBase):  # custom compiler extends from CompilerBase
    def define_pipelines(self):
        dpb = DefaultPassBuilder
        pm = PassManager("MlirTempCompiler")
        untyped_passes = dpb.define_untyped_pipeline(self.state)
        pm.passes.extend(untyped_passes.passes)

        pm.add_pass(NopythonTypeInference, "nopython frontend")
        pm.add_pass(MlirBackendInner, "mlir backend")

        pm.finalize()
        return [pm]


DEFAULT_FLAGS = Flags()
DEFAULT_FLAGS.nrt = True
setattr(DEFAULT_FLAGS, "gpu_fp64_truncate", False)
setattr(DEFAULT_FLAGS, "gpu_use_64bit_index", False)


def _compile_isolated(func, args, return_type=None, flags=DEFAULT_FLAGS, locals={}):
    typingctx = numba_mlir_target.typing_context
    targetctx = numba_mlir_target.target_context

    return compile_extra(
        typingctx,
        targetctx,
        func,
        args,
        return_type,
        flags,
        locals,
        pipeline_class=MlirTempCompiler,
    )


def compile_func(func, args, flags=DEFAULT_FLAGS):
    _compile_isolated(func, args, flags=flags)
    return get_mlir_func()
