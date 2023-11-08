# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


try:
    import dpnp

    _dnnp_available = True
except ImportError:
    _dnnp_available = False


if _dnnp_available:
    from ..numpy import funcs as numpy_funcs

    register_func = numpy_funcs.register_func

    register_func("dpnp.empty", dpnp.empty)(numpy_funcs.empty_impl)
