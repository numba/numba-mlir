# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import functools

from numba.core.untyped_passes import ReconstructSSA
from numba.core.typed_passes import NopythonTypeInference
from numba.core.compiler import (
    CompilerBase,
    DefaultPassBuilder,
    DEFAULT_FLAGS,
    compile_extra,
)
from numba.core.compiler_machinery import PassManager
from numba.core.registry import cpu_target
from numba.core import typing, cpu

from numba_mlir.mlir.passes import get_inner_backend, get_mlir_func


@functools.lru_cache
def get_temp_backend(fp64_trunc, use_64bit_index):
    backend = get_inner_backend(fp64_trunc, use_64bit_index)

    class MlirTempCompiler(CompilerBase):  # custom compiler extends from CompilerBase
        def define_pipelines(self):
            dpb = DefaultPassBuilder
            pm = PassManager("MlirTempCompiler")
            untyped_passes = dpb.define_untyped_pipeline(self.state)
            pm.passes.extend(untyped_passes.passes)

            pm.add_pass(ReconstructSSA, "ssa")
            pm.add_pass(NopythonTypeInference, "nopython frontend")
            pm.add_pass(backend, "mlir backend")

            pm.finalize()
            return [pm]

    return MlirTempCompiler


def _compile_isolated(func, args, return_type=None, flags=DEFAULT_FLAGS, locals={}):
    typingctx = cpu_target.typing_context
    targetctx = cpu_target.target_context
    fp64_truncate = getattr(flags, "fp64_truncate", False)
    use_64bit_index = getattr(flags, "use_64bit_index", True)

    pipeline = get_temp_backend(fp64_truncate, use_64bit_index)

    return compile_extra(
        typingctx,
        targetctx,
        func,
        args,
        return_type,
        flags,
        locals,
        pipeline_class=pipeline,
    )


def compile_func(func, args, flags=DEFAULT_FLAGS):
    _compile_isolated(func, args, flags=flags)
    return get_mlir_func()
