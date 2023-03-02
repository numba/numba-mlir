// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <array>
#include <cstdint>
#include <cstdlib>

#include "numba-mlir-math-runtime_export.h"

template <size_t NumDims, typename T> struct Memref {
  void *userData;
  T *data;
  intptr_t offset;
  std::array<intptr_t, NumDims> dims;
  std::array<intptr_t, NumDims> strides;
};
