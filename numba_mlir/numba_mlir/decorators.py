# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Define @jit and related decorators.
"""

import warnings

from .mlir.target import numba_mlir_jit
from .mlir.vectorize import vectorize as mlir_vectorize
from .mlir.settings import USE_MLIR

from numba.core.decorators import jit as orig_jit
from numba.core.decorators import njit as orig_njit
from numba.np.ufunc import vectorize as orig_vectorize


mlir_jit = numba_mlir_jit


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
    return mlir_jit(*args, **kws)


def mlir_njit_replace_parfors(
    signature_or_function=None,
    locals={},
    cache=False,
    pipeline_class=None,
    boundscheck=False,
    **options
):
    options.update({"nopython": True})
    pipeline = mlir_compiler_replace_parfors_pipeline
    return orig_jit(
        signature_or_function=signature_or_function,
        locals=locals,
        cache=cache,
        pipeline_class=pipeline,
        boundscheck=boundscheck,
        **options
    )


if USE_MLIR:
    jit = mlir_jit
    njit = mlir_njit
    vectorize = mlir_vectorize
    njit_replace_parfors = mlir_njit_replace_parfors
else:
    jit = orig_jit
    njit = orig_njit
    vectorize = orig_vectorize
    njit_replace_parfors = orig_njit


def override_numba_decorators():
    if USE_MLIR:
        import numba

        numba.jit = jit
        numba.njit = njit
        numba.vectorize = vectorize
