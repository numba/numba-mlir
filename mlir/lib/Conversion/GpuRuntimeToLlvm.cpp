// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "numba/Conversion/GpuRuntimeToLlvm.hpp"

#include "numba/Dialect/gpu_runtime/IR/GpuRuntimeOps.hpp"
#include "numba/Transforms/FuncUtils.hpp"
#include "numba/Transforms/TypeConversion.hpp"

#include "GpuCommon.hpp"

#include <mlir/Conversion/AsyncToLLVM/AsyncToLLVM.h>
#include <mlir/Conversion/GPUCommon/GPUCommonPass.h>
#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/Pattern.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/GPU/Transforms/Passes.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

namespace {
struct FunctionCallBuilder {
  FunctionCallBuilder(mlir::StringRef functionName, mlir::Type returnType,
                      mlir::ArrayRef<mlir::Type> argumentTypes)
      : functionName(functionName),
        functionType(
            mlir::LLVM::LLVMFunctionType::get(returnType, argumentTypes)) {}
  mlir::LLVM::CallOp create(mlir::Location loc, mlir::OpBuilder &builder,
                            mlir::ArrayRef<mlir::Value> arguments) const {
    auto module =
        builder.getBlock()->getParent()->getParentOfType<mlir::ModuleOp>();
    auto function = [&] {
      if (auto function =
              module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(functionName))
        return function;
      return mlir::OpBuilder::atBlockEnd(module.getBody())
          .create<mlir::LLVM::LLVMFuncOp>(loc, functionName, functionType);
    }();
    return builder.create<mlir::LLVM::CallOp>(loc, function, arguments);
  }

private:
  mlir::StringRef functionName;
  mlir::LLVM::LLVMFunctionType functionType;
};

static constexpr llvm::StringLiteral kEventCountAttrName("gpu.event_count");
static constexpr llvm::StringLiteral kEventIndexAttrName("gpu.event_index");

static mlir::Type getLLVMPointerType(mlir::Type elemType) {
  assert(elemType);
  return mlir::LLVM::LLVMPointerType::get(elemType.getContext());
}

template <typename OpTy>
class ConvertOpToGpuRuntimeCallPattern
    : public mlir::ConvertOpToLLVMPattern<OpTy> {
public:
  explicit ConvertOpToGpuRuntimeCallPattern(mlir::LLVMTypeConverter &converter)
      : mlir::ConvertOpToLLVMPattern<OpTy>(converter) {}

protected:
  mlir::MLIRContext *context = &this->getTypeConverter()->getContext();

  mlir::Type llvmVoidType = mlir::LLVM::LLVMVoidType::get(context);
  mlir::Type llvmPointerType =
      getLLVMPointerType(mlir::IntegerType::get(context, 8));
  mlir::Type llvmPointerPointerType = getLLVMPointerType(llvmPointerType);
  mlir::Type llvmInt8Type = mlir::IntegerType::get(context, 8);
  mlir::Type llvmInt32Type = mlir::IntegerType::get(context, 32);
  mlir::Type llvmInt64Type = mlir::IntegerType::get(context, 64);
  mlir::Type llvmIndexType = mlir::IntegerType::get(
      context, this->getTypeConverter()->getPointerBitwidth(0));

  mlir::Type llvmI32PtrType = getLLVMPointerType(llvmIndexType);

  // Must be kept in sync with gpu common header def.
  mlir::Type llvmGpuParamType = mlir::LLVM::LLVMStructType::getLiteral(
      context, {llvmPointerType, llvmInt32Type, llvmInt32Type});
  mlir::Type llvmGpuParamPointerType = getLLVMPointerType(llvmGpuParamType);
  mlir::Type llvmAllocResType = mlir::LLVM::LLVMStructType::getLiteral(
      context, {llvmPointerType, llvmPointerType, llvmPointerType});
  mlir::Type llvmAllocResPtrType = getLLVMPointerType(llvmAllocResType);

  FunctionCallBuilder streamCreateCallBuilder = {
      "gpuxStreamCreate",
      llvmPointerType, // stream
      {
          llvmIndexType,  // events count
          llvmPointerType // device name
      }};

  FunctionCallBuilder streamDestroyCallBuilder = {"gpuxStreamDestroy",
                                                  llvmVoidType,
                                                  {
                                                      llvmPointerType // stream
                                                  }};

  FunctionCallBuilder moduleLoadCallBuilder = {"gpuxModuleLoad",
                                               llvmPointerType, // module
                                               {
                                                   llvmPointerType, // stream
                                                   llvmPointerType, // data ptr
                                                   llvmIndexType,   // data size
                                               }};

  FunctionCallBuilder moduleDestroyCallBuilder = {"gpuxModuleDestroy",
                                                  llvmVoidType,
                                                  {
                                                      llvmPointerType // module
                                                  }};

  FunctionCallBuilder kernelGetCallBuilder = {"gpuxKernelGet",
                                              llvmPointerType, // kernel
                                              {
                                                  llvmPointerType, // module
                                                  llvmPointerType, // name
                                              }};

  FunctionCallBuilder kernelDestroyCallBuilder = {"gpuxKernelDestroy",
                                                  llvmVoidType,
                                                  {
                                                      llvmPointerType // kernel
                                                  }};

  FunctionCallBuilder launchKernelCallBuilder = {
      "gpuxLaunchKernel",
      llvmPointerType, // dep
      {
          llvmPointerType,        // stream
          llvmPointerType,        // kernel
          llvmIndexType,          // gridXDim
          llvmIndexType,          // gridyDim
          llvmIndexType,          // gridZDim
          llvmIndexType,          // blockXDim
          llvmIndexType,          // blockYDim
          llvmIndexType,          // blockZDim
          llvmPointerPointerType, // deps (null-term)
          llvmGpuParamPointerType,   // params (null-term)
          llvmIndexType,          // eventIndex
      }};

  FunctionCallBuilder waitEventCallBuilder = {"gpuxWait",
                                              llvmVoidType,
                                              {
                                                  llvmPointerType // dep
                                              }};

  FunctionCallBuilder allocCallBuilder = {
      "gpuxAlloc",
      llvmVoidType,
      {
          llvmPointerType,        // stream
          llvmIndexType,          // size
          llvmIndexType,          // alignment
          llvmInt32Type,          // shared
          llvmPointerPointerType, // deps (null-term)
          llvmIndexType,          // eventIndex
          llvmAllocResPtrType,    // result
      }};

  FunctionCallBuilder deallocCallBuilder = {
      "gpuxDeAlloc",
      llvmVoidType,
      {
          llvmPointerType, // stream
          llvmPointerType, // memory pointer
      }};

  FunctionCallBuilder suggestBlockSizeBuilder = {
      "gpuxSuggestBlockSize",
      llvmVoidType,
      {
          llvmPointerType, // stream
          llvmPointerType, // kernel
          llvmI32PtrType,  // grid sizes
          llvmI32PtrType,  // ret block sizes
          llvmIndexType,   // dim count
      }};

  mlir::Value createDepsArray(mlir::OpBuilder &rewriter, mlir::Location loc,
                              mlir::Operation *op,
                              mlir::ValueRange deps) const {
    auto depsArraySize = static_cast<unsigned>(deps.size());
    auto depsArrayType =
        mlir::LLVM::LLVMArrayType::get(llvmPointerType, depsArraySize + 1);
    mlir::Value depsArray =
        rewriter.create<mlir::LLVM::UndefOp>(loc, depsArrayType);
    for (auto i : llvm::seq(0u, depsArraySize)) {
      depsArray = rewriter.create<mlir::LLVM::InsertValueOp>(loc, depsArray,
                                                             deps[i], i);
    }
    auto nullPtr = rewriter.create<mlir::LLVM::NullOp>(loc, llvmPointerType);
    depsArray = rewriter.create<mlir::LLVM::InsertValueOp>(
        loc, depsArray, nullPtr, depsArraySize);

    auto depsArrayPtrType = getLLVMPointerType(depsArrayType);
    numba::AllocaInsertionPoint allocaHelper(op);
    auto depsArrayPtr = allocaHelper.insert(rewriter, [&]() {
      auto size = rewriter.create<mlir::LLVM::ConstantOp>(
          loc, llvmInt64Type, rewriter.getI64IntegerAttr(1));
      return rewriter.create<mlir::LLVM::AllocaOp>(loc, depsArrayPtrType,
                                                   depsArrayType, size, 0);
    });

    rewriter.create<mlir::LLVM::StoreOp>(loc, depsArray, depsArrayPtr);

    return rewriter.create<mlir::LLVM::BitcastOp>(loc, llvmPointerPointerType,
                                                  depsArrayPtr);
  }

  mlir::Value createEventIndexVar(mlir::OpBuilder &rewriter, mlir::Location loc,
                                  mlir::Operation *op) const {
    auto eventIndex = [&]() -> int64_t {
      auto value = mlir::getConstantIntValue(op->getAttr(kEventIndexAttrName));
      if (!value)
        return -1;

      return *value;
    }();
    return rewriter.create<mlir::LLVM::ConstantOp>(
        loc, llvmIndexType, rewriter.getIntegerAttr(llvmIndexType, eventIndex));
  }
};

class ConvertGpuStreamCreatePattern
    : public ConvertOpToGpuRuntimeCallPattern<gpu_runtime::CreateGpuStreamOp> {
public:
  ConvertGpuStreamCreatePattern(mlir::LLVMTypeConverter &converter)
      : ConvertOpToGpuRuntimeCallPattern<gpu_runtime::CreateGpuStreamOp>(
            converter) {}

private:
  mlir::LogicalResult
  matchAndRewrite(gpu_runtime::CreateGpuStreamOp op,
                  gpu_runtime::CreateGpuStreamOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    if (!mod)
      return mlir::failure();

    auto eventsCount =
        mlir::getConstantIntValue(mod->getAttr(kEventCountAttrName));
    if (!eventsCount)
      return mlir::failure();

    auto device = adaptor.getDeviceAttr().dyn_cast_or_null<mlir::StringAttr>();

    auto eventsCountAttr = rewriter.getIntegerAttr(llvmIndexType, *eventsCount);
    auto loc = op.getLoc();
    mlir::Value eventsCountVar = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, llvmIndexType, eventsCountAttr);

    mlir::Value data;
    if (device) {
      llvm::SmallString<64> name = device.getValue();
      name.push_back('\0');

      auto varName = numba::getUniqueLLVMGlobalName(mod, "device_name");
      data = mlir::LLVM::createGlobalString(
          loc, rewriter, varName, name, mlir::LLVM::Linkage::Internal, true);
    } else {
      data = rewriter.create<mlir::LLVM::NullOp>(loc, llvmPointerType);
    }

    auto res =
        streamCreateCallBuilder.create(loc, rewriter, {eventsCountVar, data});
    rewriter.replaceOp(op, res.getResults());
    return mlir::success();
  }
};

class ConvertGpuStreamDestroyPattern
    : public ConvertOpToGpuRuntimeCallPattern<gpu_runtime::DestroyGpuStreamOp> {
public:
  ConvertGpuStreamDestroyPattern(mlir::LLVMTypeConverter &converter)
      : ConvertOpToGpuRuntimeCallPattern<gpu_runtime::DestroyGpuStreamOp>(
            converter) {}

private:
  mlir::LogicalResult
  matchAndRewrite(gpu_runtime::DestroyGpuStreamOp op,
                  gpu_runtime::DestroyGpuStreamOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto res =
        streamDestroyCallBuilder.create(loc, rewriter, adaptor.getSource());
    rewriter.replaceOp(op, res.getResults());
    return mlir::success();
  }
};

class ConvertGpuModuleLoadPattern
    : public ConvertOpToGpuRuntimeCallPattern<gpu_runtime::LoadGpuModuleOp> {
public:
  ConvertGpuModuleLoadPattern(mlir::LLVMTypeConverter &converter)
      : ConvertOpToGpuRuntimeCallPattern<gpu_runtime::LoadGpuModuleOp>(
            converter) {}

private:
  mlir::LogicalResult
  matchAndRewrite(gpu_runtime::LoadGpuModuleOp op,
                  gpu_runtime::LoadGpuModuleOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    if (!mod)
      return mlir::failure();

    auto gpuMod = mod.lookupSymbol<mlir::gpu::GPUModuleOp>(op.getModule());
    if (!gpuMod)
      return mlir::failure();

    auto blobAttr = gpuMod->getAttrOfType<mlir::StringAttr>(
        mlir::gpu::getDefaultGpuBinaryAnnotation());
    if (!blobAttr)
      return mlir::failure();

    auto blob = blobAttr.getValue();

    auto loc = op.getLoc();
    auto name = numba::getUniqueLLVMGlobalName(mod, "gpu_blob");
    auto data = mlir::LLVM::createGlobalString(
        loc, rewriter, name, blob, mlir::LLVM::Linkage::Internal, true);
    auto size = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, llvmIndexType,
        mlir::IntegerAttr::get(llvmIndexType,
                               static_cast<int64_t>(blob.size())));
    auto res = moduleLoadCallBuilder.create(loc, rewriter,
                                            {adaptor.getStream(), data, size});
    rewriter.replaceOp(op, res.getResults());
    return mlir::success();
  }
};

class ConvertGpuModuleDestroyPattern
    : public ConvertOpToGpuRuntimeCallPattern<gpu_runtime::DestroyGpuModuleOp> {
public:
  ConvertGpuModuleDestroyPattern(mlir::LLVMTypeConverter &converter)
      : ConvertOpToGpuRuntimeCallPattern<gpu_runtime::DestroyGpuModuleOp>(
            converter) {}

private:
  mlir::LogicalResult
  matchAndRewrite(gpu_runtime::DestroyGpuModuleOp op,
                  gpu_runtime::DestroyGpuModuleOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto res =
        moduleDestroyCallBuilder.create(loc, rewriter, adaptor.getSource());
    rewriter.replaceOp(op, res.getResults());
    return mlir::success();
  }
};

class ConvertGpuKernelGetPattern
    : public ConvertOpToGpuRuntimeCallPattern<gpu_runtime::GetGpuKernelOp> {
public:
  ConvertGpuKernelGetPattern(mlir::LLVMTypeConverter &converter)
      : ConvertOpToGpuRuntimeCallPattern<gpu_runtime::GetGpuKernelOp>(
            converter) {}

private:
  mlir::LogicalResult
  matchAndRewrite(gpu_runtime::GetGpuKernelOp op,
                  gpu_runtime::GetGpuKernelOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    if (!mod)
      return mlir::failure();

    auto loc = op.getLoc();
    llvm::SmallString<64> name = op.getKernel().getLeafReference().getValue();
    name.push_back('\0');

    auto varName = numba::getUniqueLLVMGlobalName(mod, "kernel_name");
    auto data = mlir::LLVM::createGlobalString(
        loc, rewriter, varName, name, mlir::LLVM::Linkage::Internal, true);
    auto res =
        kernelGetCallBuilder.create(loc, rewriter, {adaptor.getModule(), data});
    rewriter.replaceOp(op, res.getResults());
    return mlir::success();
  }
};

class ConvertGpuKernelDestroyPattern
    : public ConvertOpToGpuRuntimeCallPattern<gpu_runtime::DestroyGpuKernelOp> {
public:
  ConvertGpuKernelDestroyPattern(mlir::LLVMTypeConverter &converter)
      : ConvertOpToGpuRuntimeCallPattern<gpu_runtime::DestroyGpuKernelOp>(
            converter) {}

private:
  mlir::LogicalResult
  matchAndRewrite(gpu_runtime::DestroyGpuKernelOp op,
                  gpu_runtime::DestroyGpuKernelOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto res =
        kernelDestroyCallBuilder.create(loc, rewriter, adaptor.getSource());
    rewriter.replaceOp(op, res.getResults());
    return mlir::success();
  }
};

template<unsigned Size>
static bool isInt(mlir::Type type) {
  auto intType = mlir::dyn_cast<mlir::IntegerType>(type);
  if (!intType)
    return false;

  return intType.getWidth() == Size;
}

template<unsigned Size>
static bool isFloat(mlir::Type type) {
  auto floatType = mlir::dyn_cast<mlir::FloatType>(type);
  if (!floatType)
    return false;

  return floatType.getWidth() == Size;
}

static bool isPointer(mlir::Type type) {
  return mlir::isa<mlir::LLVM::LLVMPointerType>(type);
}

static std::optional<mlir::TypedAttr> getGpuParamType(mlir::Type type) {
  assert(type);
  using CheckFuncT = bool(*)(mlir::Type);
  using PType = numba::GpuParamType;
  const std::pair<CheckFuncT, PType> handlers[] = {
    // clang-format off
    {&isInt<8>,    PType::int8},
    {&isInt<16>,   PType::int16},
    {&isInt<32>,   PType::int32},
    {&isInt<64>,   PType::int64},
    {&isFloat<32>, PType::float32},
    {&isFloat<64>, PType::float64},
    {&isPointer,   PType::ptr},
    // clang-format on
  };

  for (auto &&[handler, val]: handlers) {
    if (handler(type)) {
      auto intType = mlir::IntegerType::get(type.getContext(), 32);
      return mlir::IntegerAttr::get(intType, static_cast<int64_t>(val));
    }
  }
  return std::nullopt;
}

class ConvertGpuKernelLaunchPattern
    : public ConvertOpToGpuRuntimeCallPattern<gpu_runtime::LaunchGpuKernelOp> {
public:
  ConvertGpuKernelLaunchPattern(mlir::LLVMTypeConverter &converter)
      : ConvertOpToGpuRuntimeCallPattern<gpu_runtime::LaunchGpuKernelOp>(
            converter) {}

private:
  mlir::LogicalResult
  matchAndRewrite(gpu_runtime::LaunchGpuKernelOp op,
                  gpu_runtime::LaunchGpuKernelOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto depsArrayPtr =
        createDepsArray(rewriter, loc, op, adaptor.getAsyncDependencies());

    numba::AllocaInsertionPoint allocaHelper(op);
    auto kernelParams = adaptor.getKernelOperands();
    auto paramsCount = static_cast<unsigned>(kernelParams.size());
    auto paramsArrayType =
        mlir::LLVM::LLVMArrayType::get(llvmGpuParamType, paramsCount + 1);
    auto paramsArrayPtrType = getLLVMPointerType(paramsArrayType);

    auto getKernelParamType = [&](unsigned i) -> mlir::Type {
      if (op.getKernelOperands()[i].getType().isa<mlir::MemRefType>()) {
        mlir::MemRefDescriptor desc(kernelParams[i]);
        return desc.getElementPtrType();
      }

      return kernelParams[i].getType();
    };

    llvm::SmallVector<mlir::Value> paramsStorage(paramsCount);
    auto paramsArrayPtr = allocaHelper.insert(rewriter, [&]() {
      auto one = rewriter.create<mlir::LLVM::ConstantOp>(
          loc, llvmInt64Type, rewriter.getI64IntegerAttr(1));
      for (auto i : llvm::seq(0u, paramsCount)) {
        auto paramType = getKernelParamType(i);
        auto ptrType = getLLVMPointerType(paramType);
        paramsStorage[i] = rewriter.create<mlir::LLVM::AllocaOp>(
            loc, ptrType, paramType, one, 0);
      }
      return rewriter.create<mlir::LLVM::AllocaOp>(loc, paramsArrayPtrType,
                                                   paramsArrayType, one, 0);
    });

    mlir::Value one = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, llvmInt32Type, rewriter.getI32IntegerAttr(1));

    auto localMemStorageClass = mlir::gpu::AddressSpaceAttr::get(
        rewriter.getContext(),
        mlir::gpu::GPUDialect::getWorkgroupAddressSpace());

    auto computeTypeSize = [&](mlir::Type type) -> mlir::Value {
      // %Size = getelementptr %T* null, int 1
      // %SizeI = ptrtoint %T* %Size to i32
      auto ptrType = getLLVMPointerType(type);
      auto nullPtr = rewriter.create<mlir::LLVM::NullOp>(loc, ptrType);
      auto gep =
          rewriter.create<mlir::LLVM::GEPOp>(loc, ptrType, type, nullPtr, one);
      return rewriter.create<mlir::LLVM::PtrToIntOp>(loc, llvmInt32Type, gep);
    };

    auto getKernelParam =
        [&](unsigned i) -> std::pair<mlir::Value, mlir::Value> {
      auto memrefType =
          op.getKernelOperands()[i].getType().dyn_cast<mlir::MemRefType>();
      auto paramType = paramsStorage[i].getType();
      if (memrefType) {
        mlir::MemRefDescriptor desc(kernelParams[i]);
        if (memrefType.getMemorySpace() == localMemStorageClass) {
          auto rank = static_cast<unsigned>(memrefType.getRank());
          auto typeSize = std::max(memrefType.getElementTypeBitWidth(), 8u) / 8;
          mlir::Value size = rewriter.create<mlir::LLVM::ConstantOp>(
              loc, llvmInt32Type,
              rewriter.getIntegerAttr(llvmInt32Type, typeSize));
          for (auto i : llvm::seq(0u, rank)) {
            auto dim = desc.size(rewriter, loc, i);
            size = rewriter.create<mlir::LLVM::MulOp>(loc, llvmInt32Type, size,
                                                      dim);
          }
          return {size, nullptr};
        }
        auto size = computeTypeSize(paramType);
        return {size, desc.alignedPtr(rewriter, loc)};
      }

      auto size = computeTypeSize(paramType);
      return {size, kernelParams[i]};
    };

    mlir::Value paramsArray =
        rewriter.create<mlir::LLVM::UndefOp>(loc, paramsArrayType);

    for (auto i : llvm::seq(0u, paramsCount)) {
      auto paramType = paramsStorage[i].getType();
      auto typeAttr = getGpuParamType(paramType);
      if (!typeAttr)
        return mlir::failure();

      auto param = getKernelParam(i);
      mlir::Value ptr;
      if (!param.second) {
        ptr = rewriter.create<mlir::LLVM::NullOp>(loc, llvmPointerType);
      } else {
        rewriter.create<mlir::LLVM::StoreOp>(loc, param.second,
                                             paramsStorage[i]);
        ptr = rewriter.create<mlir::LLVM::BitcastOp>(loc, llvmPointerType,
                                                     paramsStorage[i]);
      }

      auto typeSize = param.first;

      mlir::Value range =
          rewriter.create<mlir::LLVM::UndefOp>(loc, llvmGpuParamType);
      range = rewriter.create<mlir::LLVM::InsertValueOp>(loc, range, ptr, 0);
      range =
          rewriter.create<mlir::LLVM::InsertValueOp>(loc, range, typeSize, 1);

      auto typeConst = rewriter.create<mlir::LLVM::ConstantOp>(loc, *typeAttr);
      range = rewriter.create<mlir::LLVM::InsertValueOp>(loc, range, typeConst, 2);

      paramsArray = rewriter.create<mlir::LLVM::InsertValueOp>(loc, paramsArray,
                                                               range, i);
    }

    auto nullPtr = rewriter.create<mlir::LLVM::NullOp>(loc, llvmPointerType);
    auto nullRange = [&]() {
      auto zero = rewriter.create<mlir::LLVM::ConstantOp>(
          loc, rewriter.getIntegerAttr(llvmInt32Type, 0));
      mlir::Value range =
          rewriter.create<mlir::LLVM::UndefOp>(loc, llvmGpuParamType);
      range =
          rewriter.create<mlir::LLVM::InsertValueOp>(loc, range, nullPtr, 0);
      range = rewriter.create<mlir::LLVM::InsertValueOp>(loc, range, zero, 1);

      auto nullTypeAttr = mlir::IntegerAttr::get(llvmInt32Type, static_cast<int64_t>(numba::GpuParamType::null));
      auto typeConst = rewriter.create<mlir::LLVM::ConstantOp>(loc, nullTypeAttr);
      range = rewriter.create<mlir::LLVM::InsertValueOp>(loc, range, typeConst, 2);

      return range;
    }();
    paramsArray = rewriter.create<mlir::LLVM::InsertValueOp>(
        loc, paramsArray, nullRange, paramsCount);
    rewriter.create<mlir::LLVM::StoreOp>(loc, paramsArray, paramsArrayPtr);

    auto eventIndexVar = createEventIndexVar(rewriter, loc, op);

    auto paramsArrayVoidPtr = rewriter.create<mlir::LLVM::BitcastOp>(
        loc, llvmGpuParamPointerType, paramsArrayPtr);
    mlir::Value params[] = {
        // clang-format off
        adaptor.getStream(),
        adaptor.getKernel(),
        adaptor.getGridSizeX(),
        adaptor.getGridSizeY(),
        adaptor.getGridSizeZ(),
        adaptor.getBlockSizeX(),
        adaptor.getBlockSizeY(),
        adaptor.getBlockSizeZ(),
        depsArrayPtr,
        paramsArrayVoidPtr,
        eventIndexVar,
        // clang-format on
    };
    auto event =
        launchKernelCallBuilder.create(loc, rewriter, params)->getResult(0);
    if (op.getNumResults() == 0) {
      waitEventCallBuilder.create(loc, rewriter, event);
      rewriter.eraseOp(op);
    } else {
      rewriter.replaceOp(op, event);
    }
    return mlir::success();
  }
};

class ConvertGpuAllocPattern
    : public ConvertOpToGpuRuntimeCallPattern<gpu_runtime::GPUAllocOp> {
public:
  ConvertGpuAllocPattern(mlir::LLVMTypeConverter &converter)
      : ConvertOpToGpuRuntimeCallPattern<gpu_runtime::GPUAllocOp>(converter) {}

private:
  mlir::LogicalResult
  matchAndRewrite(gpu_runtime::GPUAllocOp op,
                  gpu_runtime::GPUAllocOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (!op.getSymbolOperands().empty())
      return mlir::failure();

    auto memrefType = op.getType();
    auto converter = getTypeConverter();
    auto dstType = converter->convertType(memrefType);
    if (!dstType)
      return mlir::failure();

    bool isShared = op.getHostShared();

    auto localMemStorageClass = mlir::gpu::AddressSpaceAttr::get(
        rewriter.getContext(),
        mlir::gpu::GPUDialect::getWorkgroupAddressSpace());
    bool isLocal = memrefType.getMemorySpace() == localMemStorageClass;

    if (isShared && isLocal)
      return mlir::failure();

    auto loc = op.getLoc();

    mlir::SmallVector<mlir::Value, 4> shape;
    mlir::SmallVector<mlir::Value, 4> strides;
    mlir::Value sizeBytes;
    getMemRefDescriptorSizes(loc, memrefType, adaptor.getDynamicSizes(),
                             rewriter, shape, strides, sizeBytes);

    assert(shape.size() == strides.size());

    auto alignment = rewriter.getIntegerAttr(llvmIndexType, 64);
    auto alignmentVar =
        rewriter.create<mlir::LLVM::ConstantOp>(loc, llvmIndexType, alignment);

    auto memType = numba::GpuAllocType::Device;
    if (isShared) {
      memType = numba::GpuAllocType::Shared;
    } else if (isLocal) {
      memType = numba::GpuAllocType::Local;
    }

    auto typeVar = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, llvmInt32Type,
        rewriter.getI32IntegerAttr(static_cast<int>(memType)));

    auto depsArrayPtr =
        createDepsArray(rewriter, loc, op, adaptor.getAsyncDependencies());

    auto eventIndexVar = createEventIndexVar(rewriter, loc, op);

    numba::AllocaInsertionPoint allocaHelper(op);
    auto resultPtr = allocaHelper.insert(rewriter, [&]() {
      auto size = rewriter.create<mlir::LLVM::ConstantOp>(
          loc, llvmInt64Type, rewriter.getI64IntegerAttr(1));
      return rewriter.create<mlir::LLVM::AllocaOp>(loc, llvmAllocResPtrType,
                                                   llvmAllocResType, size, 0);
    });

    mlir::Value params[] = {
        // clang-format off
        adaptor.getStream(),
        sizeBytes,
        alignmentVar,
        typeVar,
        depsArrayPtr,
        eventIndexVar,
        resultPtr,
        // clang-format on
    };
    allocCallBuilder.create(loc, rewriter, params);
    auto res =
        rewriter.create<mlir::LLVM::LoadOp>(loc, llvmAllocResType, resultPtr);
    auto meminfo = rewriter.create<mlir::LLVM::ExtractValueOp>(
        loc, llvmPointerType, res, 0);
    auto dataPtr = rewriter.create<mlir::LLVM::ExtractValueOp>(
        loc, llvmPointerType, res, 1);

    auto memrefDesc = mlir::MemRefDescriptor::undef(rewriter, loc, dstType);
    auto elemPtrTye = memrefDesc.getElementPtrType();
    memrefDesc.setAllocatedPtr(
        rewriter, loc,
        rewriter.create<mlir::LLVM::BitcastOp>(loc, elemPtrTye, meminfo));
    memrefDesc.setAlignedPtr(
        rewriter, loc,
        rewriter.create<mlir::LLVM::BitcastOp>(loc, elemPtrTye, dataPtr));

    auto zero = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, llvmIndexType, rewriter.getIntegerAttr(llvmIndexType, 0));

    memrefDesc.setOffset(rewriter, loc, zero);
    for (auto i : llvm::seq(0u, static_cast<unsigned>(shape.size()))) {
      memrefDesc.setSize(rewriter, loc, i, shape[i]);
      memrefDesc.setStride(rewriter, loc, i, strides[i]);
    }

    mlir::Value resMemref = memrefDesc;
    mlir::Value event = rewriter.create<mlir::LLVM::ExtractValueOp>(
        loc, llvmPointerType, res, 2);
    if (op.getNumResults() == 1) {
      waitEventCallBuilder.create(loc, rewriter, event);
      rewriter.replaceOp(op, resMemref);
    } else {
      mlir::Value vals[] = {
          resMemref,
          event,
      };
      rewriter.replaceOp(op, vals);
    }
    return mlir::success();
  }
};

class ConvertGpuDeAllocPattern
    : public ConvertOpToGpuRuntimeCallPattern<gpu_runtime::GPUDeallocOp> {
public:
  ConvertGpuDeAllocPattern(mlir::LLVMTypeConverter &converter)
      : ConvertOpToGpuRuntimeCallPattern<gpu_runtime::GPUDeallocOp>(converter) {
  }

private:
  mlir::LogicalResult
  matchAndRewrite(gpu_runtime::GPUDeallocOp op,
                  gpu_runtime::GPUDeallocOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    mlir::Value pointer =
        mlir::MemRefDescriptor(adaptor.getMemref()).allocatedPtr(rewriter, loc);
    auto casted =
        rewriter.create<mlir::LLVM::BitcastOp>(loc, llvmPointerType, pointer);
    mlir::Value params[] = {adaptor.getStream(), casted};
    auto res = deallocCallBuilder.create(loc, rewriter, params);
    rewriter.replaceOp(op, res.getResults());
    return mlir::success();
  }
};

class ConvertGpuSuggestBlockSizePattern
    : public ConvertOpToGpuRuntimeCallPattern<
          gpu_runtime::GPUSuggestBlockSizeOp> {
public:
  ConvertGpuSuggestBlockSizePattern(mlir::LLVMTypeConverter &converter)
      : ConvertOpToGpuRuntimeCallPattern<gpu_runtime::GPUSuggestBlockSizeOp>(
            converter) {}

private:
  mlir::LogicalResult
  matchAndRewrite(gpu_runtime::GPUSuggestBlockSizeOp op,
                  gpu_runtime::GPUSuggestBlockSizeOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto numDims = op.getNumResults();
    auto loc = op.getLoc();
    numba::AllocaInsertionPoint allocaHelper(op);
    auto gridArrayPtr = allocaHelper.insert(rewriter, [&]() {
      auto size = rewriter.create<mlir::LLVM::ConstantOp>(
          loc, llvmInt64Type, rewriter.getI64IntegerAttr(numDims));
      return rewriter.create<mlir::LLVM::AllocaOp>(loc, llvmI32PtrType,
                                                   llvmInt32Type, size, 0);
    });
    auto blockArrayPtr = allocaHelper.insert(rewriter, [&]() {
      auto size = rewriter.create<mlir::LLVM::ConstantOp>(
          loc, llvmInt64Type, rewriter.getI64IntegerAttr(numDims));
      return rewriter.create<mlir::LLVM::AllocaOp>(loc, llvmI32PtrType,
                                                   llvmInt32Type, size, 0);
    });

    auto sizesType = mlir::LLVM::LLVMArrayType::get(llvmInt32Type, numDims);
    auto sizesPtrType = getLLVMPointerType((sizesType));
    auto castToSizesPtrType = [&](mlir::Value val) {
      return rewriter.create<mlir::LLVM::BitcastOp>(loc, sizesPtrType, val);
    };

    mlir::Value gridArray =
        rewriter.create<mlir::LLVM::UndefOp>(loc, sizesType);
    for (auto i : llvm::seq(0u, numDims)) {
      auto gridSize = rewriter.create<mlir::LLVM::TruncOp>(
          loc, llvmInt32Type, adaptor.getGridSize()[i]);
      gridArray = rewriter.create<mlir::LLVM::InsertValueOp>(loc, gridArray,
                                                             gridSize, i);
    }

    rewriter.create<mlir::LLVM::StoreOp>(loc, gridArray,
                                         castToSizesPtrType(gridArrayPtr));
    mlir::Value numDimsVal = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, llvmIndexType, rewriter.getIntegerAttr(llvmIndexType, numDims));

    mlir::Value params[] = {
        // clang-format off
        adaptor.getStream(),
        adaptor.getKernel(),
        gridArrayPtr,
        blockArrayPtr,
        numDimsVal,
        // clang-format on
    };

    suggestBlockSizeBuilder.create(loc, rewriter, params);

    mlir::Value blockSizeArray = rewriter.create<mlir::LLVM::LoadOp>(
        loc, sizesType, castToSizesPtrType(blockArrayPtr));
    llvm::SmallVector<mlir::Value, 3> result(numDims);
    for (auto i : llvm::seq(0u, numDims)) {
      auto blockSize = rewriter.create<mlir::LLVM::ExtractValueOp>(
          loc, llvmInt32Type, blockSizeArray, i);
      result[i] =
          rewriter.create<mlir::LLVM::ZExtOp>(loc, llvmIndexType, blockSize);
    }

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct EnumerateEventsPass
    : public mlir::PassWrapper<EnumerateEventsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EnumerateEventsPass)

  void runOnOperation() override {
    auto mod = getOperation();
    int64_t eventCount = 0;
    auto *ctx = &getContext();
    auto intType = mlir::IntegerType::get(ctx, 64);
    auto indexAttrName = mlir::StringAttr::get(ctx, kEventIndexAttrName);
    auto countAttrName = mlir::StringAttr::get(ctx, kEventCountAttrName);
    mod.walk([&](mlir::gpu::AsyncOpInterface op) {
      op->setAttr(indexAttrName, mlir::IntegerAttr::get(intType, eventCount));
      ++eventCount;
    });
    mod->setAttr(countAttrName, mlir::IntegerAttr::get(intType, eventCount));
  }
};

struct GPUToLLVMPass
    : public mlir::PassWrapper<GPUToLLVMPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GPUToLLVMPass)

  void runOnOperation() override {
    mlir::MLIRContext &context = getContext();
    mlir::LLVMTypeConverter converter(&context);
    mlir::RewritePatternSet patterns(&context);
    mlir::LLVMConversionTarget target(context);

    mlir::populateAsyncStructuralTypeConversionsAndLegality(converter, patterns,
                                                            target);
    mlir::populateGpuToLLVMConversionPatterns(
        converter, patterns, mlir::gpu::getDefaultGpuBinaryAnnotation());

    numba::populateControlFlowTypeConversionRewritesAndTarget(converter,
                                                              patterns, target);

    gpu_runtime::populateGpuToLLVMPatternsAndLegality(converter, patterns,
                                                      target);

    auto mod = getOperation();
    if (mlir::failed(
            mlir::applyPartialConversion(mod, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

// Expose the passes to the outside world
std::unique_ptr<mlir::Pass> gpu_runtime::createEnumerateEventsPass() {
  return std::make_unique<EnumerateEventsPass>();
}

void gpu_runtime::populateGpuToLLVMPatternsAndLegality(
    mlir::LLVMTypeConverter &converter, mlir::RewritePatternSet &patterns,
    mlir::ConversionTarget &target) {
  auto context = patterns.getContext();
  auto llvmPointerType = getLLVMPointerType(mlir::IntegerType::get(context, 8));
  converter.addConversion(
      [llvmPointerType](gpu_runtime::OpaqueType) -> mlir::Type {
        return llvmPointerType;
      });
  converter.addConversion(
      [llvmPointerType](gpu_runtime::StreamType) -> mlir::Type {
        return llvmPointerType;
      });

  converter.addTypeAttributeConversion(
      [](mlir::BaseMemRefType type,
         mlir::gpu::AddressSpaceAttr /*memorySpaceAttr*/) -> mlir::IntegerAttr {
        auto ctx = type.getContext();
        return mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), 0);
      });

  patterns.insert<
      // clang-format off
      ConvertGpuStreamCreatePattern,
      ConvertGpuStreamDestroyPattern,
      ConvertGpuModuleLoadPattern,
      ConvertGpuModuleDestroyPattern,
      ConvertGpuKernelGetPattern,
      ConvertGpuKernelDestroyPattern,
      ConvertGpuKernelLaunchPattern,
      ConvertGpuAllocPattern,
      ConvertGpuDeAllocPattern,
      ConvertGpuSuggestBlockSizePattern
      // clang-format on
      >(converter);

  target.addIllegalDialect<mlir::gpu::GPUDialect>();
  target.addIllegalDialect<gpu_runtime::GpuRuntimeDialect>();
}

std::unique_ptr<mlir::Pass> gpu_runtime::createGPUToLLVMPass() {
  return std::make_unique<GPUToLLVMPass>();
}
