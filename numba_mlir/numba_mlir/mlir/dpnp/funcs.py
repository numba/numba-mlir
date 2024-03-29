# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


try:
    import dpnp

    _dnnp_available = True
except:
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

    def _gen_unary_ops():
        ops = [
            "sqrt",
            "square",
            "log",
            "sin",
            "cos",
            "exp",
            "tanh",
            "abs",
            "negative",
            # "positive",
            "logical_not",
            "invert",
        ]
        for op_name in ops:
            fn_name = "dpnp." + op_name
            func = getattr(dpnp, op_name)
            impl_name = op_name + "_impl"
            impl = getattr(numpy_funcs, impl_name)
            register_func(op_name, func)(impl)

    _gen_unary_ops()
    del _gen_unary_ops

    def _gen_binary_ops():
        ops = [
            "logical_and",
            "logical_or",
            "logical_xor",
            "bitwise_and",
            "bitwise_or",
            "bitwise_xor",
            "left_shift",
            "right_shift",
        ]
        for op_name in ops:
            fn_name = "dpnp." + op_name
            func = getattr(dpnp, op_name)
            impl_name = op_name + "_impl"
            impl = getattr(numpy_funcs, impl_name)
            register_func(op_name, func)(impl)

    _gen_binary_ops()
    del _gen_binary_ops
