# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Define @jit and related decorators.
"""

from .mlir.compiler import (
    mlir_compiler_pipeline,
    get_gpu_pipeline,
    mlir_compiler_replace_parfors_pipeline,
)
from .mlir.vectorize import vectorize as mlir_vectorize
from .mlir.settings import USE_MLIR

from numba.core.decorators import jit as orig_jit
from numba.core.decorators import njit as orig_njit
from numba.np.ufunc import vectorize as orig_vectorize


def mlir_jit(
    signature_or_function=None,
    locals={},
    cache=False,
    pipeline_class=None,
    boundscheck=False,
    **options
):
    if not options.get("nopython", False):
        return orig_jit(
            signature_or_function=signature_or_function,
            locals=locals,
            cache=cache,
            boundscheck=boundscheck,
            **options
        )

    fp64_truncate = options.get("gpu_fp64_truncate", False)
    assert fp64_truncate in [
        True,
        False,
        "auto",
    ], 'gpu_fp64_truncate supported values are True/False/"auto"'
    options.pop("gpu_fp64_truncate", None)

    use_64bit_index = options.get("gpu_use_64bit_index", True)
    assert use_64bit_index in [
        True,
        False,
    ], "gpu_use_64bit_index supported values are True/False"
    options.pop("gpu_use_64bit_index", None)

    if options.get("replace_parfors", False):
        pipeline = mlir_compiler_replace_parfors_pipeline
    elif options.get("enable_gpu_pipeline", True):
        pipeline = get_gpu_pipeline(fp64_truncate, use_64bit_index)
    else:
        pipeline = mlir_compiler_pipeline

    options.pop("replace_parfors", None)
    options.pop("enable_gpu_pipeline", None)
    options.pop(
        "access_types", None
    )  # pop them to ignore since they are not a part of numba but dppy.
    return orig_jit(
        signature_or_function=signature_or_function,
        locals=locals,
        cache=cache,
        pipeline_class=pipeline,
        boundscheck=boundscheck,
        **options
    )


def mlir_njit(*args, **kws):
    """
    Equivalent to jit(nopython=True)

    See documentation for jit function/decorator for full description.
    """
    if "nopython" in kws:
        warnings.warn("nopython is set for njit and is ignored", RuntimeWarning)
    if "forceobj" in kws:
        warnings.warn("forceobj is set for njit and is ignored", RuntimeWarning)
        del kws["forceobj"]
    kws.update({"nopython": True})
    return jit(*args, **kws)


if USE_MLIR:
    jit = mlir_jit
    njit = mlir_njit
    vectorize = mlir_vectorize
else:
    jit = orig_jit
    njit = orig_njit
    vectorize = orig_vectorize


def override_numba_decorators():
    if USE_MLIR:
        import numba

        numba.jit = jit
        numba.njit = njit
        numba.vectorize = vectorize
