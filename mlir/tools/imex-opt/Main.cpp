// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/IR/MLIRContext.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

#include "imex/Dialect/gpu_runtime/IR/GpuRuntimeOps.hpp"
#include "imex/Dialect/imex_util/Dialect.hpp"
#include "imex/Dialect/ntensor/IR/NTensorOps.hpp"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<gpu_runtime::GpuRuntimeDialect>();
  registry.insert<numba::ntensor::NTensorDialect>();
  registry.insert<numba::util::ImexUtilDialect>();
  return mlir::failed(MlirOptMain(argc, argv, "imex modular optimizer driver\n",
                                  registry,
                                  /*preloadDialectsInContext=*/false));
}
