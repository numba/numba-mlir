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

    _init_impl = numpy_funcs._init_impl

    register_func = numpy_funcs.register_func

    @register_func("dpnp.empty", dpnp.empty)
    def empty_impl(
        builder, shape, dtype=None, layout="C", device=None, usm_type="device"
    ):
        return _init_impl(builder, shape, dtype)

    register_func("dpnp.sin", dpnp.sin)(numpy_funcs.sin_impl)
