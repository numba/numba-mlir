// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef GPURUNTIME_OPS
#define GPURUNTIME_OPS

include "mlir/Dialect/GPU/IR/GPUBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

def GpuRuntime_Dialect : Dialect {
  let name = "gpu_runtime";
  let description = [{
      GpuRuntime Dialect for representing Gpu runtime ops for level zero runtime
    }];
  let cppNamespace = "::gpu_runtime";

  let dependentDialects = [
    "::numba::util::NumbaUtilDialect"
  ];

  let hasCanonicalizer = 1;
  let hasConstantMaterializer = 1;
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
}

class GpuRuntime_Type<string name, string typeMnemonic, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Type">
    : TypeDef<GpuRuntime_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

def GpuRuntime_OpaqueType : GpuRuntime_Type<"Opaque", "opaque", [], "::mlir::Type">;

def GpuRuntime_QueueType : GpuRuntime_Type<"Queue", "queue", [], "::mlir::Type">;

class GpuRuntime_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<GpuRuntime_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

class GpuRuntime_Op<string mnemonic, list<Trait> traits = []> :
      Op<GpuRuntime_Dialect, mnemonic, traits>;

def GPURegionDescAttr
    : GpuRuntime_Attr<"GPURegionDesc", "region_desc"> {
  let parameters = (ins
    "::mlir::StringAttr":$device,
    "::mlir::StringAttr":$usm_type, // TODO: enum
    "int16_t":$spirv_major_version,
    "int16_t":$spirv_minor_version,
    "bool":$has_fp16,
    "bool":$has_fp64);
  let assemblyFormat = "`<` struct(params) `>`";

  let builders = [
    AttrBuilderWithInferredContext<(ins "::mlir::MLIRContext *":$context,
                                        "::mlir::StringRef":$device,
                                        "::mlir::StringRef":$usm_type,
                                        "int16_t":$spirv_major_version,
                                        "int16_t":$spirv_minor_version,
                                        "bool":$has_fp16,
                                        "bool":$has_fp64), [{
      mlir::OpBuilder builder(context);
      auto devAttr = builder.getStringAttr(device);
      auto usmAttr = builder.getStringAttr(usm_type);
      return $_get(context, devAttr, usmAttr, spirv_major_version, spirv_minor_version, has_fp16, has_fp64);
    }]>
  ];
}

def CreateGpuQueueOp : GpuRuntime_Op<"create_gpu_queue", [Pure]> {
  let arguments = (ins OptionalAttr<AnyAttr>:$device);
  let results = (outs GpuRuntime_QueueType:$result);

  let hasCanonicalizer = 1;

  let builders = [OpBuilder<(ins CArg<"::mlir::Attribute", "{}">:$device)>];

  let assemblyFormat = "attr-dict ( `,` $device^)?";
}

def DestroyGpuQueueOp : GpuRuntime_Op<"destroy_gpu_queue"> {
  let arguments = (ins GpuRuntime_QueueType:$source);

  let assemblyFormat = "attr-dict $source";
}

def LoadGpuModuleOp : GpuRuntime_Op<"load_gpu_module", [Pure]> {
  let arguments = (ins GpuRuntime_QueueType:$queue, SymbolRefAttr:$module);
  let results = (outs GpuRuntime_OpaqueType:$result);

  let hasCanonicalizer = 1;

  let builders = [OpBuilder<(ins "::mlir::Value"
                             : $queue, "::mlir::gpu::GPUModuleOp"
                             : $module)>];
}

def DestroyGpuModuleOp : GpuRuntime_Op<"destroy_gpu_module"> {
  let arguments = (ins GpuRuntime_OpaqueType : $source);
}

def GetGpuKernelOp : GpuRuntime_Op<"get_gpu_kernel", [Pure]> {
  let arguments = (ins GpuRuntime_OpaqueType : $module, SymbolRefAttr : $kernel);
  let results = (outs GpuRuntime_OpaqueType : $result);

  let hasCanonicalizer = 1;

  let builders = [OpBuilder<(ins "::mlir::Value"
                             : $module, "::mlir::gpu::GPUFuncOp"
                             : $kernel)>];
}

def LaunchGpuKernelOp
    : GpuRuntime_Op<"launch_gpu_kernel",
               [GPU_AsyncOpInterface, AttrSizedOperandSegments]> {
  let arguments = (ins Variadic<GPU_AsyncToken>:$asyncDependencies,
                     GpuRuntime_QueueType:$queue,
                     GpuRuntime_OpaqueType:$kernel,
                     Index:$gridSizeX, Index:$gridSizeY, Index:$gridSizeZ,
                     Index:$blockSizeX, Index:$blockSizeY, Index:$blockSizeZ,
                     Variadic<AnyType>:$kernelOperands);
  let results = (outs Optional<GPU_AsyncToken> : $asyncToken);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins "::mlir::Value" : $queue,
                                 "::mlir::Value" : $kernel,
                                 "::mlir::gpu::KernelDim3" : $gridSize,
                                 "::mlir::gpu::KernelDim3" : $blockSize,
                                 "::mlir::ValueRange" : $kernelOperands)>];
}

def DestroyGpuKernelOp : GpuRuntime_Op<"destroy_gpu_kernel"> {
  let arguments = (ins GpuRuntime_OpaqueType : $source);
}

def GPUAllocOp : GpuRuntime_Op<"alloc",
  [GPU_AsyncOpInterface, AttrSizedOperandSegments]> {

  let arguments = (ins Variadic<GPU_AsyncToken>:$asyncDependencies,
                       GpuRuntime_QueueType:$queue,
                       Variadic<Index>:$dynamicSizes,
                       Variadic<Index>:$symbolOperands,
                       UnitAttr:$hostShared);
  let results = (outs Res<AnyMemRef, "", [MemAlloc]>:$memref,
                 Optional<GPU_AsyncToken>:$asyncToken);

  let extraClassDeclaration = [{
    ::mlir::MemRefType getType() { return getMemref().getType().cast<::mlir::MemRefType>(); }
  }];

  let assemblyFormat = [{ $queue
    custom<AsyncDependencies>(type($asyncToken), $asyncDependencies) (` ` `host_shared` $hostShared^)? ` `
    `(` $dynamicSizes `)` (`` `[` $symbolOperands^ `]`)? attr-dict `:` type($memref)
  }];
}

def GPUDeallocOp : GpuRuntime_Op<"dealloc"> {
  let arguments = (ins Variadic<GPU_AsyncToken>:$asyncDependencies,
                  Res<AnyMemRef, "", [MemAlloc]>:$memref,
                  GpuRuntime_QueueType:$queue);

  let results = (outs Optional<GPU_AsyncToken>:$asyncToken);

  let assemblyFormat = [{ $queue
    custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
    $memref attr-dict `:` type($memref)
  }];
}

def GPUSuggestBlockSizeOp : GpuRuntime_Op<"suggest_block_size",
                                     [AttrSizedOperandSegments, Pure]> {
  let arguments = (ins Optional<GpuRuntime_QueueType>:$queue,
                       Optional<GpuRuntime_OpaqueType>:$kernel,
                       OptionalAttr<SymbolRefAttr>:$kernelRef,
                       Variadic<Index>:$gridSize);

  let results = (outs Variadic<Index>:$results);

  let builders = [
    OpBuilder<(ins
      "::std::optional<::mlir::Value>":$queue,
      "::mlir::ValueRange":$gridSize,
      "::mlir::Value":$kernel)>,
    OpBuilder<(ins
      "::std::optional<::mlir::Value>":$queue,
      "::mlir::ValueRange":$gridSize,
      "::mlir::SymbolRefAttr":$kernel)>,
    OpBuilder<(ins
      "::std::optional<::mlir::Value>":$queue,
      "::mlir::ValueRange":$gridSize)>
  ];

  let extraClassDeclaration = [{
      /// The name of the kernel's containing module.
      ::mlir::StringAttr getKernelModuleName();

      /// The name of the kernel.
      ::mlir::StringAttr getKernelName();
  }];

  let assemblyFormat = [{
    attr-dict (`:` $queue^)? ($kernel^)? ($kernelRef^)? `,` $gridSize `->` type($results)
  }];
}

def GPUMemFenceOp : GpuRuntime_Op<"mem_fence"> {
  let summary = "Orders loads and stores of a work-item executing a kernel.";

  let arguments = (ins I64Attr:$flags);

  let assemblyFormat = "$flags attr-dict";
}

def GPUGlobalReduceOp : GpuRuntime_Op<"global_reduce", [
    IsolatedFromAbove,
    SingleBlockImplicitTerminator<"::gpu_runtime::GPUGlobalReduceYieldOp">,
    TypesMatchWith<"type of 'value' matches element type of 'target'",
                   "target", "value",
                   "$_self.cast<::mlir::MemRefType>().getElementType()">
  ]> {
  let summary = "Reduce values among global exectuion space.";
  let description = [{
    The `global_reduce` op reduces the value of every work item. The result is
    equal for all work items.
  }];

  let arguments = (ins AnyType:$value, Arg<AnyMemRef, "", [MemWrite]>:$target);

  let regions = (region AnyRegion:$region);
  let assemblyFormat = "attr-dict $value `,` $target `:` type($target) $region";
}

def GPUGlobalReduceYieldOp : GpuRuntime_Op<"global_reduce_yield", [
    Pure,
    ReturnLike,
    Terminator,
    HasParent<"::gpu_runtime::GPUGlobalReduceOp">
  ]> {

  let summary = "Global reduce yield and termination operation";
  let description = [{
    "global_reduce_yield" yields an SSA value from the "global_reduce" op region
    and terminates it.
  }];

  let arguments = (ins AnyType:$result);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

  let assemblyFormat = "attr-dict $result `:` type($result)";
}

#endif // GPURUNTIME_OPS
