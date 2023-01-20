# SPDX-FileCopyrightText: 2021 - 2023 Intel Corporation
# SPDX-FileCopyrightText: 2023 Numba project
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .decorators import *

from .mlir.settings import DPNP_AVAILABLE

from . import _version

__version__ = _version.get_versions()["version"]
