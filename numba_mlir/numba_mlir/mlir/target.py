# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from numba.core import types
from numba.core.typing import Context
from numba.core.typing.templates import Registry
from numba.extending import typeof_impl as numba_typeof_impl
from numba.core.typing.typeof import Purpose, _TypeofContext, _termcolor

from functools import singledispatch


def typeof(val, purpose=Purpose.argument):
    """
    Get the Numba type of a Python value for the given purpose.
    """
    # Note the behaviour for Purpose.argument must match _typeof.c.
    c = _TypeofContext(purpose)
    ty = typeof_impl(val, c)
    if ty is None:
        msg = _termcolor.errmsg(f"Cannot determine Numba type of {type(val)}")
        raise ValueError(msg)
    return ty


@singledispatch
def typeof_impl(val, c):
    """
    Use Numba impl by default
    """
    return numba_typeof_impl(val, c)


registry = Registry()
infer = registry.register
infer_global = registry.register_global
infer_getattr = registry.register_attr


class NumbaMLIRContext(Context):
    def load_additional_registries(self):
        self.install_registry(registry)
        super().load_additional_registries()

    def resolve_argument_type(self, val):
        """
        Return the numba type of a Python value that is being used
        as a function argument.  Integer types will all be considered
        int64, regardless of size.

        ValueError is raised for unsupported types.
        """
        return typeof(val, Purpose.argument)

    def resolve_value_type(self, val):
        """
        Return the numba type of a Python value that is being used
        as a runtime constant.
        ValueError is raised for unsupported types.
        """
        try:
            ty = typeof(val, Purpose.constant)
        except ValueError as e:
            # Make sure the exception doesn't hold a reference to the user
            # value.
            typeof_exc = utils.erase_traceback(e)
        else:
            return ty

        if isinstance(val, types.ExternalFunction):
            return val

        # Try to look up target specific typing information
        ty = self._get_global_type(val)
        if ty is not None:
            return ty

        raise typeof_exc
