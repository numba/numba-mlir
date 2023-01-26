# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Define compiler pipelines.
"""

from .lowering import mlir_NativeLowering

import functools


from numba_mlir.mlir.passes import (
    MlirDumpPlier,
    MlirBackend,
    get_gpu_backend,
    MlirReplaceParfors,
)
from numba.core.compiler_machinery import PassManager
from numba.core.compiler import CompilerBase as orig_CompilerBase
from numba.core.compiler import DefaultPassBuilder as orig_DefaultPassBuilder
from numba.core.typed_passes import NativeLowering as orig_NativeLowering
from numba.core.typed_passes import NativeParforLowering as orig_NativeParforLowering
from numba.core.typed_passes import (
    NopythonTypeInference,
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
    def define_nopython_pipeline(
        state, enable_gpu_pipeline, fp64_truncate, use_64bit_index
    ):
        pm = orig_DefaultPassBuilder.define_nopython_pipeline(state, "nopython")

        import numba_mlir.mlir.settings

        if numba_mlir.mlir.settings.USE_MLIR:
            if enable_gpu_pipeline:
                pm.add_pass_after(
                    get_gpu_backend(fp64_truncate, use_64bit_index),
                    NopythonTypeInference,
                )
            else:
                pm.add_pass_after(MlirBackend, NopythonTypeInference)
            pm.passes, replaced = _replace_pass(
                pm.passes, orig_NativeLowering, mlir_NativeLowering
            )
            if replaced == 0:
                pm.passes, replaced = _replace_pass(
                    pm.passes, orig_NativeParforLowering, mlir_NativeLowering
                )
            assert replaced == 1, "Failed to replace lowering pass"

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
            pm.add_pass_after(MlirDumpPlier, NopythonTypeInference)

        pm.finalize()
        return pm

    def define_replace_parfors_pipeline(state, name="nopython"):
        pm = orig_DefaultPassBuilder.define_nopython_pipeline(state, name)

        import numba_mlir.mlir.settings

        if numba_mlir.mlir.settings.USE_MLIR:
            pm.add_pass_after(MlirReplaceParfors, ParforPass)

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


@functools.lru_cache
def get_gpu_pipeline(fp64_truncate, use_64bit_index):
    class mlir_compiler_gpu_pipeline(orig_CompilerBase):
        def define_pipelines(self):
            # this maintains the objmode fallback behaviour
            pms = []
            if not self.state.flags.force_pyobject:
                pms.append(
                    mlir_PassBuilder.define_nopython_pipeline(
                        self.state,
                        enable_gpu_pipeline=True,
                        fp64_truncate=fp64_truncate,
                        use_64bit_index=use_64bit_index,
                    )
                )
            if self.state.status.can_fallback or self.state.flags.force_pyobject:
                pms.append(mlir_PassBuilder.define_objectmode_pipeline(self.state))
            return pms

    return mlir_compiler_gpu_pipeline


class mlir_compiler_replace_parfors_pipeline(orig_CompilerBase):
    def define_pipelines(self):
        # this maintains the objmode fallback behaviour
        pms = []
        if not self.state.flags.force_pyobject:
            pms.append(mlir_PassBuilder.define_replace_parfors_pipeline(self.state))
        if self.state.status.can_fallback or self.state.flags.force_pyobject:
            pms.append(mlir_PassBuilder.define_objectmode_pipeline(self.state))
        return pms
