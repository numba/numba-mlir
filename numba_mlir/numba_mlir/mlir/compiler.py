# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Define compiler pipelines.
"""

from .lowering import mlir_NativeLowering

import functools

from numba.core.typed_passes import AnnotateTypes, IRLegalization

from numba_mlir.mlir.passes import MlirDumpPlier, MlirBackend, get_gpu_backend
from numba.core.compiler_machinery import PassManager
from numba.core.compiler import CompilerBase as orig_CompilerBase
from numba.core.compiler import DefaultPassBuilder as orig_DefaultPassBuilder
from numba.core.typed_passes import NativeLowering as orig_NativeLowering
from numba.core.typed_passes import (
    PreParforPass,
    ParforPass,
    DumpParforDiagnostics,
    NopythonRewrites,
    PreLowerStripPhis,
    InlineOverloads,
    NopythonRewrites,
    IRLegalization,
)


def _replace_pass(passes, old_pass, new_pass):
    count = 0
    ret = []
    for p, n in passes:
        if p == old_pass:
            count += 1
            ret.append((new_pass, str(new_pass)))
        else:
            ret.append((p, n))
    return ret, count


def _remove_passes(passes, to_remove):
    count = 0
    ret = []
    for p, n in passes:
        if p in to_remove:
            count += 1
        else:
            ret.append((p, n))
    return ret, count


class mlir_PassBuilder(orig_DefaultPassBuilder):
    @staticmethod
    def define_nopython_pipeline(state, enable_gpu_pipeline, fp64_truncate):
        pm = orig_DefaultPassBuilder.define_nopython_pipeline(state, "nopython")

        import numba_mlir.mlir.settings

        if numba_mlir.mlir.settings.USE_MLIR:
            if enable_gpu_pipeline:
                pm.add_pass_after(get_gpu_backend(fp64_truncate), AnnotateTypes)
            else:
                pm.add_pass_after(MlirBackend, AnnotateTypes)
            pm.passes, replaced = _replace_pass(
                pm.passes, orig_NativeLowering, mlir_NativeLowering
            )
            assert replaced == 1, replaced

            pm.passes, removed = _remove_passes(
                pm.passes,
                [
                    PreParforPass,
                    ParforPass,
                    DumpParforDiagnostics,
                    NopythonRewrites,
                    PreLowerStripPhis,
                    InlineOverloads,
                    NopythonRewrites,
                    IRLegalization,
                ],
            )

        if numba_mlir.mlir.settings.DUMP_PLIER:
            pm.add_pass_after(MlirDumpPlier, AnnotateTypes)

        pm.finalize()
        return pm


class mlir_compiler_pipeline(orig_CompilerBase):
    def define_pipelines(self):
        # this maintains the objmode fallback behaviour
        pms = []
        if not self.state.flags.force_pyobject:
            pms.append(mlir_PassBuilder.define_nopython_pipeline(self.state))
        if self.state.status.can_fallback or self.state.flags.force_pyobject:
            pms.append(mlir_PassBuilder.define_objectmode_pipeline(self.state))
        return pms

@functools.cache
def get_gpu_pipeline(fp64_truncate):
    class mlir_compiler_gpu_pipeline(orig_CompilerBase):
        def define_pipelines(self):
            # this maintains the objmode fallback behaviour
            pms = []
            if not self.state.flags.force_pyobject:
                pms.append(
                    mlir_PassBuilder.define_nopython_pipeline(
                        self.state, enable_gpu_pipeline=True, fp64_truncate=fp64_truncate
                    )
                )
            if self.state.status.can_fallback or self.state.flags.force_pyobject:
                pms.append(mlir_PassBuilder.define_objectmode_pipeline(self.state))
            return pms

    return mlir_compiler_gpu_pipeline
