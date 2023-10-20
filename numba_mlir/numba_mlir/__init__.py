# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .decorators import *

from . import _version

__version__ = _version.get_versions()["version"]
