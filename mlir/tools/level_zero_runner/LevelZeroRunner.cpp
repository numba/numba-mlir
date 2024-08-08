// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/TargetSelect.h>
#include <mlir/Conversion/LLVMCommon/LoweringOptions.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/GPU/Transforms/Passes.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/Transforms/RequestCWrappers.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVDialect.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVOps.h>
#include <mlir/Dialect/SPIRV/Transforms/Passes.h>
#include <mlir/ExecutionEngine/JitRunner.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>

#include "numba/Conversion/GpuRuntimeToLlvm.hpp"
#include "numba/Conversion/GpuToGpuRuntime.hpp"
#include "numba/Conversion/UtilToLlvm.hpp"

#include "legacy/Dialect/Arith/Transforms/Passes.h"
#include "legacy/Dialect/Bufferization/Transforms/Passes.h"
#include "legacy/Dialect/Linalg/Transforms/Passes.h"
#include "legacy/Dialect/Tensor/Transforms/Passes.h"

using namespace mlir;

static LogicalResult runMLIRPasses(mlir::Operation *op,
                                   mlir::JitRunnerOptions & /*options*/) {
  auto module = mlir::cast<mlir::ModuleOp>(op);
  PassManager passManager(module.getContext());
  if (failed(applyPassManagerCLOptions(passManager)))
    return mlir::failure();

  passManager.addPass(mlir::arith::legacy::createConstantBufferizePass());
  passManager.addNestedPass<mlir::func::FuncOp>(createSCFBufferizePass());
  passManager.addNestedPass<mlir::func::FuncOp>(
      bufferization::createEmptyTensorToAllocTensorPass());
  passManager.addNestedPass<mlir::func::FuncOp>(
      mlir::linalg::legacy::createLinalgBufferizePass());
  passManager.addNestedPass<mlir::func::FuncOp>(
      bufferization::legacy::createBufferizationBufferizePass());
  passManager.addNestedPass<mlir::func::FuncOp>(
      mlir::tensor::legacy::createTensorBufferizePass());
  passManager.addPass(func::createFuncBufferizePass());
  passManager.addNestedPass<mlir::func::FuncOp>(
      bufferization::createFinalizingBufferizePass());
  // passManager.addNestedPass<mlir::func::FuncOp>(
  //     bufferization::createBufferDeallocationPass());
  passManager.addNestedPass<mlir::func::FuncOp>(
      createConvertLinalgToParallelLoopsPass());
  passManager.addNestedPass<mlir::func::FuncOp>(
      createGpuMapParallelLoopsPass());
  passManager.addNestedPass<mlir::func::FuncOp>(createParallelLoopToGpuPass());

  //  passManager.addNestedPass<mlir::func::FuncOp>(
  //      gpu_runtime::createInsertGPUAllocsPass());
  passManager.addPass(mlir::createCanonicalizerPass());
  passManager.addNestedPass<mlir::func::FuncOp>(
      mlir::createGpuDecomposeMemrefsPass());
  passManager.addNestedPass<mlir::func::FuncOp>(mlir::createLowerAffinePass());

  passManager.addPass(createGpuKernelOutliningPass());
  //  passManager.addPass(memref::createFoldSubViewOpsPass());
  passManager.addNestedPass<mlir::gpu::GPUModuleOp>(
      gpu_runtime::createAbiAttrsPass());
  passManager.addPass(gpu_runtime::createSetSPIRVCapabilitiesPass());

  passManager.addPass(gpu_runtime::createGPUToSpirvPass());
  OpPassManager &modulePM = passManager.nest<spirv::ModuleOp>();
  modulePM.addPass(spirv::createSPIRVLowerABIAttributesPass());
  modulePM.addPass(spirv::createSPIRVUpdateVCEPass());
  LowerToLLVMOptions llvmOptions(module.getContext(), DataLayout(module));
  passManager.nest<func::FuncOp>().addPass(LLVM::createRequestCWrappersPass());

  // Gpu -> GpuRuntime
  passManager.addPass(gpu_runtime::createSerializeSPIRVPass());
  passManager.addNestedPass<mlir::func::FuncOp>(gpu_runtime::createGPUExPass());

  // GpuRuntime -> LLVM

  ConvertFuncToLLVMPassOptions llvmPassOptions;
  passManager.addPass(createConvertFuncToLLVMPass(llvmPassOptions));
  passManager.addPass(gpu_runtime::createGPUToLLVMPass());
  passManager.addPass(
      numba::createUtilToLLVMPass([&](MLIRContext &) { return llvmOptions; }));
  passManager.addPass(mlir::memref::createExpandStridedMetadataPass());
  passManager.addPass(mlir::createLowerAffinePass());
  passManager.addPass(createFinalizeMemRefToLLVMConversionPass());
  passManager.addPass(createReconcileUnrealizedCastsPass());

  return passManager.run(module);
}

int main(int argc, char **argv) {
  llvm::llvm_shutdown_obj x;
  registerPassManagerCLOptions();

  llvm::InitLLVM y(argc, argv);
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  mlir::JitRunnerConfig jitRunnerConfig;
  jitRunnerConfig.mlirTransformer = runMLIRPasses;

  mlir::DialectRegistry registry;
  registry.insert<mlir::cf::ControlFlowDialect, mlir::arith::ArithDialect,
                  mlir::LLVM::LLVMDialect, mlir::gpu::GPUDialect,
                  mlir::spirv::SPIRVDialect, mlir::func::FuncDialect,
                  mlir::memref::MemRefDialect, mlir::linalg::LinalgDialect,
                  mlir::tensor::TensorDialect>();
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerBuiltinDialectTranslation(registry);

  return mlir::JitRunnerMain(argc, argv, registry, jitRunnerConfig);
}
