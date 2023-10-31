// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <pybind11/pybind11.h>

#include <llvm/ADT/StringMap.h>
#include <llvm/TargetParser/Host.h>

#include "PyModule.hpp"

#include "Lowering.hpp"

static bool isMKLSupported() {
#ifdef NUMBA_MLIR_USE_MKL
  return true;
#else
  return false;
#endif
}

static bool isSyclMKLSupported() {
#ifdef NUMBA_MLIR_USE_SYCL_MKL
  return true;
#else
  return false;
#endif
}

static unsigned getVectorLength() {
  llvm::StringMap<bool, llvm::MallocAllocator> features;
  if (!llvm::sys::getHostCPUFeatures(features))
    return 128;

  auto checkFlag = [&](llvm::StringRef name) -> bool {
    auto it = features.find(name);
    return it != features.end() && it->second;
  };

  if (checkFlag("avx512f"))
    return 512;

  if (checkFlag("avx2"))
    return 256;

  return 128;
}

PYBIND11_MODULE(mlir_compiler, m) {
  m.def("init_compiler", &initCompiler, "No docs");
  m.def("create_module", &createModule, "No docs");
  m.def("lower_function", &lowerFunction, "No docs");
  m.def("lower_parfor", &lowerParfor, "No docs");
  m.def("compile_module", &compileModule, "No docs");
  m.def("register_symbol", &registerSymbol, "No docs");
  m.def("get_function_pointer", &getFunctionPointer, "No docs");
  m.def("release_module", &releaseModule, "No docs");
  m.def("module_str", &moduleStr, "No docs");
  m.def("is_mkl_supported", &isMKLSupported, "No docs");
  m.def("is_sycl_mkl_supported", &isSyclMKLSupported, "No docs");
  m.def("get_vector_length", &getVectorLength, "No docs");
}
