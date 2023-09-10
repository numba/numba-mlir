# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


from numba.core.typing import Context
from numba.core.typing.templates import Registry


registry = Registry()
infer = registry.register
infer_global = registry.register_global
infer_getattr = registry.register_attr


class NumbaMLIRContext(Context):
    def load_additional_registries(self):
        self.install_registry(registry)
        super().load_additional_registries()
