// SPDX-FileCopyrightText: 2021 - 2023 Intel Corporation
// SPDX-FileCopyrightText: 2023 Numba project
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cstdlib>

#include "dpcomp-python-runtime_export.h"

using MemInfoDtorFunction = void (*)(void *ptr, size_t size, void *info);

extern "C" DPCOMP_PYTHON_RUNTIME_EXPORT void *
dpcompAllocMemInfo(void *data, size_t size, MemInfoDtorFunction dtor,
                   void *dtorInfo);
