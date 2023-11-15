// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "numba/Conversion/GpuToGpuRuntime.hpp"

#include "GpuCommon.hpp"

#include "numba/Dialect/gpu_runtime/IR/GpuRuntimeOps.hpp"
#include "numba/Dialect/numba_util/Dialect.hpp"
#include "numba/Dialect/numba_util/Utils.hpp"
#include "numba/Transforms/FuncUtils.hpp"
#include "numba/Transforms/ScalarOpsConversion.hpp"
#include "numba/Transforms/TypeConversion.hpp"

#include <mlir/Conversion/ArithToSPIRV/ArithToSPIRV.h>
#include <mlir/Conversion/ControlFlowToSPIRV/ControlFlowToSPIRV.h>
#include <mlir/Conversion/FuncToSPIRV/FuncToSPIRV.h>
#include <mlir/Conversion/GPUToSPIRV/GPUToSPIRV.h>
#include <mlir/Conversion/MathToSPIRV/MathToSPIRV.h>
#include <mlir/Conversion/MemRefToSPIRV/MemRefToSPIRV.h>
#include <mlir/Conversion/SCFToSPIRV/SCFToSPIRV.h>
#include <mlir/Conversion/UBToSPIRV/UBToSPIRV.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/GPU/Transforms/ParallelLoopMapper.h>
#include <mlir/Dialect/GPU/Transforms/Passes.h>
#include <mlir/Dialect/Linalg/Utils/Utils.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVDialect.h>
#include <mlir/Dialect/SPIRV/IR/SPIRVOps.h>
#include <mlir/Dialect/SPIRV/IR/TargetAndABI.h>
#include <mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Target/SPIRV/Serialization.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/ADT/SmallBitVector.h>
#include <llvm/Support/FormatVariadic.h>

namespace {
static gpu_runtime::GPURegionDescAttr getGpuRegionEnv(mlir::Operation *op) {
  assert(op && "Invalid op");
  while (auto region =
             op->getParentOfType<numba::util::EnvironmentRegionOp>()) {
    if (auto env = mlir::dyn_cast<gpu_runtime::GPURegionDescAttr>(
            region.getEnvironment()))
      return env;

    op = region;
  }
  return {};
}

static mlir::gpu::Processor getProcessor(unsigned val) {
  const mlir::gpu::Processor mapping[] = {
      mlir::gpu::Processor::BlockX,  mlir::gpu::Processor::BlockY,
      mlir::gpu::Processor::BlockZ,  mlir::gpu::Processor::ThreadX,
      mlir::gpu::Processor::ThreadY, mlir::gpu::Processor::ThreadZ,
  };
  if (val >= std::size(mapping))
    return mlir::gpu::Processor::Sequential;

  return mapping[val];
}
struct ParallelLoopGPUMappingPass
    : public mlir::PassWrapper<ParallelLoopGPUMappingPass,
                               mlir::OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ParallelLoopGPUMappingPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::scf::SCFDialect>();
  }

  void runOnOperation() override {
    auto func = getOperation();
    auto attrName =
        mlir::StringAttr::get(&getContext(), mlir::gpu::getMappingAttrName());
    func->walk([&](numba::util::EnvironmentRegionOp envOp) {
      if (!mlir::isa<gpu_runtime::GPURegionDescAttr>(envOp.getEnvironment()))
        return;

      auto &region = envOp.getRegion();

      mlir::OpBuilder builder(&getContext());
      auto identityMap = builder.getDimIdentityMap();
      llvm::SmallVector<mlir::gpu::ParallelLoopDimMappingAttr> mapping;
      auto visitor = [&](mlir::scf::ParallelOp parallel) -> mlir::WalkResult {
        if (parallel->hasAttr(attrName))
          return mlir::WalkResult::advance();

        auto offset = [&]() -> unsigned {
          auto parent = parallel->getParentOfType<mlir::scf::ParallelOp>();
          if (!parent)
            return 0;

          auto attr = parent->getAttrOfType<mlir::ArrayAttr>(attrName);
          if (!attr || attr.empty())
            return 0;

          auto last = mlir::dyn_cast<mlir::gpu::ParallelLoopDimMappingAttr>(
              attr.getValue().back());
          if (!last)
            return 0;

          return static_cast<unsigned>(last.getProcessor()) + 1;
        }();

        auto numLoops = parallel.getNumLoops();
        mapping.resize(numLoops);
        for (auto i : llvm::seq(0u, numLoops))
          mapping[i] = builder.getAttr<mlir::gpu::ParallelLoopDimMappingAttr>(
              getProcessor(i + offset), identityMap, identityMap);

        if (mlir::failed(mlir::gpu::setMappingAttr(parallel, mapping))) {
          parallel->emitError("Failed to set mapping atter");
          return mlir::WalkResult::interrupt();
        }

        return mlir::WalkResult::advance();
      };
      if (region.walk(visitor).wasInterrupted())
        return signalPassFailure();
    });
  }
};

struct InsertGPUAllocs
    : public mlir::PassWrapper<InsertGPUAllocs,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InsertGPUAllocs)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::gpu::GPUDialect>();
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<mlir::scf::SCFDialect>();
  }

  void runOnOperation() override {
    auto func = getOperation();
    auto mod = func->getParentOfType<mlir::ModuleOp>();
    assert(mod);

    auto &funcBody = func.getBody();
    if (funcBody.empty()) {
      return;
    } else if (!llvm::hasSingleElement(funcBody)) {
      func.emitError("Function must have exactly one block");
      signalPassFailure();
      return;
    }

    struct AccessType {
      mlir::Attribute env;
      bool hostRead = false;
      bool hostWrite = false;
      bool deviceRead = false;
      bool deviceWrite = false;
    };

    llvm::SmallMapVector<mlir::Operation *, AccessType, 8> gpuBufferAllocs;
    llvm::SmallMapVector<unsigned, AccessType, 8> gpuBufferParams;
    auto &aliases = getAnalysis<mlir::BufferViewFlowAnalysis>();

    auto getMemref = [](mlir::Operation *op)
        -> std::optional<mlir::SmallVector<mlir::Value, 4>> {
      if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(op)) {
        return {{load.getMemref()}};
      } else if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(op)) {
        return {{store.getMemref()}};
      } else if (auto atomic = mlir::dyn_cast<mlir::memref::AtomicRMWOp>(op)) {
        return {{atomic.getMemref()}};
      } else if (auto call = mlir::dyn_cast<mlir::func::CallOp>(op)) {
        mlir::SmallVector<mlir::Value, 4> ret;
        for (auto arg : call.getOperands()) {
          if (mlir::isa<mlir::MemRefType>(arg.getType()))
            ret.emplace_back(arg);
        }
        return std::move(ret);
      } else {
        op->emitError("Uhhandled mem op in gpu region");
        return std::nullopt;
      }
    };

    auto hasMemAccess = [](mlir::Operation *op) -> bool {
      if (auto memInterface =
              mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op)) {
        if (memInterface.hasEffect<mlir::MemoryEffects::Read>() ||
            memInterface.hasEffect<mlir::MemoryEffects::Write>())
          return true;
      }
      if (auto call = mlir::dyn_cast<mlir::func::CallOp>(op)) {
        for (auto arg : call.getOperands()) {
          if (mlir::isa<mlir::MemRefType>(arg.getType()))
            return true;
        }
      }
      return false;
    };

    auto gpuAccessibleArg = [&]() -> llvm::SmallVector<bool> {
      auto gpuAttr = func->getAttrOfType<mlir::ArrayAttr>(
          gpu_runtime::getGpuAccessibleAttrName());
      if (!gpuAttr)
        return {};

      auto range = gpuAttr.getAsValueRange<mlir::BoolAttr>();
      return {range.begin(), range.end()};
    }();

    auto isGpuAccessibleArg = [&](unsigned i) {
      if (gpuAccessibleArg.empty())
        return false;

      assert(i < gpuAccessibleArg.size());
      return gpuAccessibleArg[i];
    };

    if (func.walk([&](mlir::Operation *op) {
              if (!op->getParentOfType<mlir::gpu::LaunchOp>())
                return mlir::WalkResult::advance();

              if (!hasMemAccess(op))
                return mlir::WalkResult::advance();

              auto memref = getMemref(op);
              if (!memref)
                return mlir::WalkResult::interrupt();

              for (auto mem : *memref) {
                while (auto parentView =
                           mem.getDefiningOp<mlir::ViewLikeOpInterface>())
                  mem = parentView.getViewSource();

                for (auto alias : aliases.resolve(mem)) {
                  auto op = alias.getDefiningOp();
                  if (op) {
                    if (mlir::isa<mlir::scf::SCFDialect>(op->getDialect()) ||
                        mlir::isa<mlir::ViewLikeOpInterface,
                                  mlir::arith::SelectOp, mlir::func::CallOp,
                                  numba::util::EnvironmentRegionOp>(op))
                      // Ignore Op
                      continue;
                    if (mlir::isa<mlir::memref::AllocOp, mlir::memref::AllocaOp,
                                  mlir::memref::GetGlobalOp>(op)) {
                      if (!op->getParentOfType<mlir::gpu::LaunchOp>())
                        gpuBufferAllocs.insert({op, {}});
                    } else {
                      op->emitError("Unhandled memref producer");
                      return mlir::WalkResult::interrupt();
                    }

                  } else {
                    auto block = alias.getParentBlock();
                    auto blockArgs = block->getArguments();
                    auto it = llvm::find(blockArgs, alias);
                    assert(it != blockArgs.end());
                    auto index = static_cast<unsigned>(it - blockArgs.begin());
                    if (!isGpuAccessibleArg(index))
                      gpuBufferParams.insert({index, {}});
                  }
                }
              }

              return mlir::WalkResult::advance();
            })
            .wasInterrupted()) {
      signalPassFailure();
      return;
    }

    auto getEnv = [](mlir::Operation *op) -> mlir::FailureOr<mlir::Attribute> {
      auto env = getGpuRegionEnv(op);
      if (!env)
        return mlir::failure();

      return env;
    };

    mlir::StringAttr devFuncAttr;
    auto isDeviceFuncCall = [&](mlir::func::CallOp call) -> bool {
      if (call->getParentOfType<mlir::gpu::LaunchOp>())
        return true;

      auto funcName = call.getCallee();
      auto origFunc = mod.lookupSymbol<mlir::func::FuncOp>(funcName);
      if (!origFunc) {
        call->emitError("Cannot resolve callee symbol");
        signalPassFailure();
        return false;
      }

      if (!devFuncAttr)
        devFuncAttr = mlir::StringAttr::get(
            &getContext(), gpu_runtime::getDeviceFuncAttrName());

      return origFunc->hasAttr(devFuncAttr);
    };

    auto getAccessType =
        [&](mlir::Value memref) -> mlir::FailureOr<AccessType> {
      AccessType ret;
      for (auto mem : aliases.resolve(memref)) {
        for (auto user : mem.getUsers()) {
          if (mlir::isa<mlir::func::ReturnOp>(user)) {
            ret.hostRead = true;
            ret.hostWrite = true;
            continue;
          }

          if (auto copy = mlir::dyn_cast<mlir::memref::CopyOp>(user)) {
            if (copy.getSource() == mem)
              ret.hostRead = true;

            if (copy.getTarget() == mem)
              ret.hostWrite = true;

            continue;
          }

          if (auto memInterface =
                  mlir::dyn_cast<mlir::MemoryEffectOpInterface>(user)) {
            bool onDevice = user->getParentOfType<mlir::gpu::LaunchOp>();
            if (memInterface.hasEffect<mlir::MemoryEffects::Read>())
              (onDevice ? ret.deviceRead : ret.hostRead) = true;

            if (memInterface.hasEffect<mlir::MemoryEffects::Write>())
              (onDevice ? ret.deviceWrite : ret.hostWrite) = true;

            if (onDevice) {
              auto env = getEnv(user);
              if (mlir::succeeded(env)) {
                assert(*env && "Invalid device");
                if (!ret.env) {
                  ret.env = *env;
                } else if (ret.env != *env) {
                  return user->emitError("Device conflict: ")
                         << ret.env << " and " << *env;
                }
              }
            }

            continue;
          }

          if (auto call = mlir::dyn_cast<mlir::func::CallOp>(user)) {
            bool onDevice = isDeviceFuncCall(call);
            (onDevice ? ret.deviceRead : ret.hostRead) = true;
            (onDevice ? ret.deviceWrite : ret.hostWrite) = true;

            if (onDevice) {
              auto env = getEnv(user);
              if (mlir::succeeded(env)) {

                assert(*env && "Invalid device");
                if (!ret.env) {
                  ret.env = *env;
                } else if (ret.env != *env) {
                  return user->emitError("Device conflict: ")
                         << ret.env << " and " << *env;
                }
              }
            }

            continue;
          }
        }
      }
      return ret;
    };

    for (auto &it : gpuBufferAllocs) {
      auto op = it.first;
      assert(op->getNumResults() == 1);
      auto access = getAccessType(op->getResult(0));
      if (mlir::failed(access))
        return signalPassFailure();

      it.second = *access;
      if (mlir::isa<mlir::memref::GetGlobalOp>(op))
        it.second.hostWrite = true;
    }

    auto &block = funcBody.front();
    for (auto &it : gpuBufferParams) {
      auto param = block.getArgument(it.first);
      auto access = getAccessType(param);
      if (mlir::failed(access))
        return signalPassFailure();

      it.second = *access;

      it.second.hostRead = true;
      it.second.hostWrite = true;
    }

    auto term = block.getTerminator();
    assert(term);

    llvm::SmallVector<mlir::Value> dims;
    llvm::SmallPtrSet<mlir::Operation *, 8> filter;
    mlir::OpBuilder builder(func);
    auto createGpuAlloc = [&](mlir::Value src, const AccessType &access) {
      auto loc = src.getLoc();
      filter.clear();
      dims.clear();
      auto memrefType = src.getType().cast<mlir::MemRefType>();
      auto rank = static_cast<unsigned>(memrefType.getRank());
      for (auto i : llvm::seq(0u, rank)) {
        if (memrefType.isDynamicDim(i)) {
          auto dimOp = builder.create<mlir::memref::DimOp>(loc, src, i);
          dims.push_back(dimOp);
          filter.insert(dimOp);
        }
      }

      auto allocType = memrefType;
      if (!allocType.getLayout().isIdentity())
        allocType = mlir::MemRefType::get(
            allocType.getShape(), allocType.getElementType(),
            mlir::MemRefLayoutAttrInterface{}, allocType.getMemorySpace());

      bool hostShared = access.hostRead || access.hostWrite;
      auto results = numba::util::wrapEnvRegion(
                         builder, src.getLoc(), access.env, memrefType,
                         [&](mlir::OpBuilder &b, mlir::Location loc) {
                           auto gpuAlloc = builder.create<mlir::gpu::AllocOp>(
                               loc, allocType, /*asyncToken*/ nullptr,
                               /*asyncDependencies*/ std::nullopt, dims,
                               /*symbolOperands*/ std::nullopt, hostShared);
                           mlir::Value allocResult = gpuAlloc.getMemref();
                           if (allocType != memrefType)
                             allocResult = builder.create<mlir::memref::CastOp>(
                                 loc, memrefType, allocResult);

                           if (access.hostWrite && access.deviceRead) {
                             auto copy = builder.create<mlir::memref::CopyOp>(
                                 loc, src, allocResult);
                             filter.insert(copy);
                           }
                           return allocResult;
                         })
                         .front();

      src.replaceAllUsesExcept(results, filter);

      builder.setInsertionPoint(term);
      numba::util::wrapEnvRegion(
          builder, src.getLoc(), access.env, std::nullopt,
          [&](mlir::OpBuilder &b, mlir::Location loc) {
            if (access.hostRead && access.deviceWrite)
              builder.create<mlir::memref::CopyOp>(loc, results, src);

            builder.create<mlir::gpu::DeallocOp>(loc, std::nullopt, results);
            return std::nullopt;
          });
    };

    for (auto &&[op, access] : gpuBufferAllocs) {
      if (auto alloc = mlir::dyn_cast<mlir::memref::AllocOp>(op)) {
        builder.setInsertionPoint(alloc);
        bool hostShared = access.hostRead || access.hostWrite;
        auto results = numba::util::wrapEnvRegion(
            builder, op->getLoc(), access.env, alloc.getType(),
            [&](mlir::OpBuilder &b, mlir::Location loc) {
              auto gpuAlloc = builder.create<mlir::gpu::AllocOp>(
                  loc, alloc.getType(), /*asyncToken*/ nullptr,
                  /*asyncDependencies*/ std::nullopt, alloc.getDynamicSizes(),
                  alloc.getSymbolOperands(), hostShared);
              return gpuAlloc.getResults();
            });
        alloc->replaceAllUsesWith(results);
        alloc.erase();
      } else if (auto alloca = mlir::dyn_cast<mlir::memref::AllocaOp>(op)) {
        alloca->emitError("Alloca is not supported yet");
        return signalPassFailure();
      } else if (auto getGlobal =
                     mlir::dyn_cast<mlir::memref::GetGlobalOp>(op)) {
        builder.setInsertionPointAfter(getGlobal);
        createGpuAlloc(getGlobal.getResult(), access);
      } else {
        llvm_unreachable("Invalid alloc type");
      }
    }

    for (auto &&[i, access] : gpuBufferParams) {
      auto param = block.getArgument(i);
      builder.setInsertionPointToStart(&block);
      createGpuAlloc(param, access);
    }
  }
};

struct ConvertGPUDeallocsPass
    : public mlir::PassWrapper<ConvertGPUDeallocsPass,
                               mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertGPUDeallocsPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::gpu::GPUDialect>();
    registry.insert<mlir::memref::MemRefDialect>();
  }

  void runOnOperation() override {
    auto op = getOperation();

    mlir::OpBuilder builder(&getContext());
    op->walk([&](mlir::gpu::DeallocOp dealloc) {
      if (dealloc.getAsyncToken()) {
        dealloc->emitError("Cannot convert gpu.dealloc with async tokens");
        signalPassFailure();
        return;
      }
      builder.setInsertionPoint(dealloc);
      builder.create<mlir::memref::DeallocOp>(dealloc->getLoc(),
                                              dealloc.getMemref());
      dealloc->erase();
    });
  }
};

static std::optional<mlir::Value> getGpuQueue(mlir::OpBuilder &builder,
                                              mlir::Operation *op) {
  assert(op);
  auto func = op->getParentOfType<mlir::FunctionOpInterface>();
  if (!func)
    return std::nullopt;

  if (!llvm::hasSingleElement(func.getFunctionBody()))
    return std::nullopt;

  mlir::Attribute device;
  if (auto env = getGpuRegionEnv(op))
    device = env.getDevice();

  auto &block = func.getFunctionBody().front();
  auto ops = block.getOps<gpu_runtime::CreateGpuQueueOp>();
  for (auto queueOp : ops)
    if (queueOp.getDeviceAttr() == device)
      return queueOp.getResult();

  mlir::OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToStart(&block);
  auto loc = builder.getUnknownLoc();
  mlir::Value queue =
      builder.create<gpu_runtime::CreateGpuQueueOp>(loc, device);
  builder.setInsertionPoint(block.getTerminator());
  builder.create<gpu_runtime::DestroyGpuQueueOp>(loc, queue);
  return queue;
}

template <typename Op>
class ConvertBitcastOp : public mlir::OpConversionPattern<Op> {
public:
  using mlir::OpConversionPattern<Op>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto converter = this->getTypeConverter();
    assert(converter && "Invalid type converter");

    auto resType = converter->convertType(op.getResult().getType());
    if (!resType)
      return mlir::failure();

    auto src = adaptor.getSource();
    auto srcType = src.getType();
    if (srcType == resType) {
      rewriter.replaceOp(op, src);
      return mlir::success();
    }

    rewriter.replaceOpWithNewOp<mlir::spirv::BitcastOp>(op, resType, src);
    return mlir::success();
  }
};

template <typename Op>
static mlir::Value lowerIntAtomic(mlir::OpBuilder &builder, mlir::Location loc,
                                  mlir::Value ptr, mlir::Value val,
                                  mlir::spirv::Scope scope) {
  return builder.create<Op>(
      loc, ptr, scope, mlir::spirv::MemorySemantics::SequentiallyConsistent,
      val);
}

template <typename Op>
static mlir::Value lowerFloatAtomic(mlir::OpBuilder &builder,
                                    mlir::Location loc, mlir::Value ptr,
                                    mlir::Value val, mlir::spirv::Scope scope) {
  return builder.create<Op>(
      loc, val.getType(), ptr, scope,
      mlir::spirv::MemorySemantics::SequentiallyConsistent, val);
}

struct ConvertAtomicRMW
    : public mlir::OpConversionPattern<mlir::memref::AtomicRMWOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::AtomicRMWOp op,
                  mlir::memref::AtomicRMWOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (!llvm::all_of(op.getIndices(),
                      [](auto v) { return mlir::isConstantIntValue(v, 0); }))
      return mlir::failure();

    auto mem = adaptor.getMemref();
    auto memType = mlir::dyn_cast<mlir::spirv::PointerType>(mem.getType());
    if (!mem)
      return mlir::failure();

    auto storageClass = memType.getStorageClass();
    auto scope = mlir::spirv::StorageClass::Workgroup == storageClass
                     ? mlir::spirv::Scope::Workgroup
                     : mlir::spirv::Scope::Device;

    auto val = adaptor.getValue();

    using func_t =
        mlir::Value (*)(mlir::OpBuilder &, mlir::Location, mlir::Value,
                        mlir::Value, mlir::spirv::Scope);

    using RMWK = mlir::arith::AtomicRMWKind;
    const std::pair<RMWK, func_t> handlers[] = {
        {RMWK::addi, &lowerIntAtomic<mlir::spirv::AtomicIAddOp>},
        {RMWK::addf, &lowerFloatAtomic<mlir::spirv::EXTAtomicFAddOp>},
    };

    auto kind = adaptor.getKind();
    for (auto &&[k, h] : handlers) {
      if (k == kind) {
        mlir::Value res = h(rewriter, op.getLoc(), mem, val, scope);
        if (!res)
          continue;

        rewriter.replaceOp(op, res);
        return mlir::success();
      }
    }
    return mlir::failure();
  }
};

static bool isBoolScalarOrVector(mlir::Type type) {
  assert(type && "Not a valid type");
  if (type.isInteger(1))
    return true;

  if (auto vecType = mlir::dyn_cast<mlir::VectorType>(type))
    return vecType.getElementType().isInteger(1);

  return false;
}

static mlir::LogicalResult
getTypeConversionFailure(mlir::ConversionPatternRewriter &rewriter,
                         mlir::Operation *op, mlir::Type srcType) {
  return rewriter.notifyMatchFailure(
      op->getLoc(),
      llvm::formatv("failed to convert source type '{0}'", srcType));
}

static mlir::LogicalResult
getTypeConversionFailure(mlir::ConversionPatternRewriter &rewriter,
                         mlir::Operation *op) {
  assert(op->getNumResults() == 1);
  return getTypeConversionFailure(rewriter, op, op->getResultTypes().front());
}

// TODO: upstream
struct ConvertI1SIndexCast
    : public mlir::OpConversionPattern<mlir::arith::IndexCastOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::arith::IndexCastOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value operand = adaptor.getIn();
    if (!isBoolScalarOrVector(operand.getType()))
      return mlir::failure();

    mlir::Location loc = op.getLoc();
    mlir::Type dstType = getTypeConverter()->convertType(op.getType());
    if (!dstType)
      return getTypeConversionFailure(rewriter, op);

    mlir::Value allOnes;
    if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(dstType)) {
      unsigned componentBitwidth = intTy.getWidth();
      allOnes = rewriter.create<mlir::spirv::ConstantOp>(
          loc, intTy,
          rewriter.getIntegerAttr(intTy,
                                  llvm::APInt::getAllOnes(componentBitwidth)));
    } else if (auto vectorTy = mlir::dyn_cast<mlir::VectorType>(dstType)) {
      unsigned componentBitwidth = vectorTy.getElementTypeBitWidth();
      allOnes = rewriter.create<mlir::spirv::ConstantOp>(
          loc, vectorTy,
          mlir::SplatElementsAttr::get(
              vectorTy, llvm::APInt::getAllOnes(componentBitwidth)));
    } else {
      return rewriter.notifyMatchFailure(
          loc, llvm::formatv("unhandled type: {0}", dstType));
    }

    mlir::Value zero = mlir::spirv::ConstantOp::getZero(dstType, loc, rewriter);
    rewriter.replaceOpWithNewOp<mlir::spirv::SelectOp>(op, dstType, operand,
                                                       allOnes, zero);
    return mlir::success();
  }
};

// TODO: upstream
struct ConvertI1UIndexCast
    : public mlir::OpConversionPattern<mlir::arith::IndexCastUIOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::arith::IndexCastUIOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Type srcType = adaptor.getOperands().front().getType();
    if (!isBoolScalarOrVector(srcType))
      return mlir::failure();

    mlir::Type dstType = getTypeConverter()->convertType(op.getType());
    if (!dstType)
      return getTypeConversionFailure(rewriter, op);

    mlir::Location loc = op.getLoc();
    mlir::Value zero = mlir::spirv::ConstantOp::getZero(dstType, loc, rewriter);
    mlir::Value one = mlir::spirv::ConstantOp::getOne(dstType, loc, rewriter);
    rewriter.replaceOpWithNewOp<mlir::spirv::SelectOp>(
        op, dstType, adaptor.getOperands().front(), one, zero);
    return mlir::success();
  }
};

// TODO: use upstream memref conversion
/// Returns true if the allocations of memref `type` generated from `allocOp`
/// can be lowered to SPIR-V.
static bool isAllocationSupported(mlir::Operation *allocOp,
                                  mlir::MemRefType type) {
  if (mlir::isa<mlir::memref::AllocOp, mlir::memref::DeallocOp>(allocOp)) {
    auto sc = mlir::dyn_cast_or_null<mlir::spirv::StorageClassAttr>(
        type.getMemorySpace());
    if (!sc || sc.getValue() != mlir::spirv::StorageClass::Workgroup)
      return false;
  } else if (mlir::isa<mlir::memref::AllocaOp>(allocOp)) {
    auto sc = mlir::dyn_cast_or_null<mlir::gpu::AddressSpaceAttr>(
        type.getMemorySpace());
    if (!sc || sc.getValue() != mlir::gpu::GPUDialect::getPrivateAddressSpace())
      return false;
  } else {
    return false;
  }

  // Currently only support static shape.
  return type.hasStaticShape();
}

/// Converts memref.alloca to SPIR-V Function variables.
class AllocaOpPattern final
    : public mlir::OpConversionPattern<mlir::memref::AllocaOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::AllocaOp allocaOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::MemRefType memrefType = allocaOp.getType();
    if (!isAllocationSupported(allocaOp, memrefType))
      return rewriter.notifyMatchFailure(allocaOp, "unhandled allocation type");

    assert(getTypeConverter() && "Invalid type converter");
    const mlir::TypeConverter &converter = *getTypeConverter();

    auto elemType = converter.convertType(memrefType.getElementType());
    if (!elemType)
      return rewriter.notifyMatchFailure(allocaOp, [&](mlir::Diagnostic &diag) {
        diag << "unsupported element type " << memrefType.getElementType();
      });

    auto resType = mlir::dyn_cast_if_present<mlir::spirv::PointerType>(
        converter.convertType(memrefType));
    if (!resType)
      return rewriter.notifyMatchFailure(allocaOp, "unhandled return type");

    auto count = memrefType.getNumElements();
    mlir::Type arrayType =
        (count == 1 ? elemType : mlir::spirv::ArrayType::get(elemType, count));
    auto allocType =
        mlir::spirv::PointerType::get(arrayType, resType.getStorageClass());

    auto loc = allocaOp.getLoc();
    mlir::Value res = rewriter.create<mlir::spirv::VariableOp>(
        loc, allocType, mlir::spirv::StorageClass::Function,
        /*initializer=*/nullptr);

    if (res.getType() != resType)
      res = rewriter.create<mlir::spirv::BitcastOp>(loc, resType, res);

    rewriter.replaceOp(allocaOp, res);
    return mlir::success();
  }
};

static gpu_runtime::FenceFlags getOpFlags(mlir::Operation *op) {
  assert(op);
  auto attr = op->getAttrOfType<mlir::IntegerAttr>(
      gpu_runtime::getFenceFlagsAttrName());
  if (!attr)
    return gpu_runtime::FenceFlags::global;

  return static_cast<gpu_runtime::FenceFlags>(attr.getValue().getSExtValue());
}

static std::optional<mlir::spirv::MemorySemantics>
getSpirvMemSematics(gpu_runtime::FenceFlags flags) {
  if (flags == gpu_runtime::FenceFlags::global) {
    return mlir::spirv::MemorySemantics::SequentiallyConsistent |
           mlir::spirv::MemorySemantics::CrossWorkgroupMemory;
  }
  if (flags == gpu_runtime::FenceFlags::local) {
    return mlir::spirv::MemorySemantics::SequentiallyConsistent |
           mlir::spirv::MemorySemantics::WorkgroupMemory;
  }
  return std::nullopt;
}

class ConvertBarrierOp
    : public mlir::OpConversionPattern<mlir::gpu::BarrierOp> {
public:
  // Set benefit higher than upstream lowering.
  ConvertBarrierOp(mlir::TypeConverter &typeConverter,
                   mlir::MLIRContext *context)
      : mlir::OpConversionPattern<mlir::gpu::BarrierOp>(typeConverter, context,
                                                        /*benefit*/ 10) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::gpu::BarrierOp op,
                  mlir::gpu::BarrierOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto flags = getOpFlags(op);
    auto semantics = getSpirvMemSematics(flags);
    if (!semantics)
      return mlir::failure();

    auto scope = mlir::spirv::Scope::Workgroup;
    rewriter.replaceOpWithNewOp<mlir::spirv::ControlBarrierOp>(op, scope, scope,
                                                               *semantics);
    return mlir::success();
  }
};

class ConvertMemFenceOp
    : public mlir::OpConversionPattern<gpu_runtime::GPUMemFenceOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(gpu_runtime::GPUMemFenceOp op,
                  gpu_runtime::GPUMemFenceOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto flags = static_cast<gpu_runtime::FenceFlags>(adaptor.getFlags());
    auto semantics = getSpirvMemSematics(flags);
    if (!semantics)
      return mlir::failure();

    auto scope = mlir::spirv::Scope::Workgroup;
    rewriter.replaceOpWithNewOp<mlir::spirv::MemoryBarrierOp>(op, scope,
                                                              *semantics);
    return mlir::success();
  }
};

static std::optional<mlir::spirv::StorageClass>
convertStorageClass(mlir::Attribute src) {
  if (auto attr = mlir::dyn_cast_or_null<mlir::gpu::AddressSpaceAttr>(src)) {
    if (attr.getValue() == mlir::gpu::GPUDialect::getWorkgroupAddressSpace())
      return mlir::spirv::StorageClass::Workgroup;

    if (attr.getValue() == mlir::gpu::GPUDialect::getPrivateAddressSpace())
      return mlir::spirv::StorageClass::Function;
  }

  return std::nullopt;
}

static mlir::spirv::StorageClass
convertStorageClass(mlir::Attribute src, mlir::spirv::StorageClass def) {
  auto ret = convertStorageClass(src);
  if (ret)
    return *ret;

  return def;
}

class ConvertGlobalOp
    : public mlir::OpConversionPattern<mlir::memref::GlobalOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::GlobalOp op,
                  mlir::memref::GlobalOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto memrefType = op.getType();
    if (!memrefType.hasStaticShape())
      return mlir::failure();

    auto storageClass = convertStorageClass(memrefType.getMemorySpace());
    if (!storageClass)
      return mlir::failure();

    auto converter = getTypeConverter();
    assert(converter);

    auto elemType = converter->convertType(memrefType.getElementType());
    if (!elemType)
      return mlir::failure();

    auto elemCount = memrefType.getNumElements();
    auto newType = mlir::spirv::ArrayType::get(elemType, elemCount);
    auto ptrType = mlir::spirv::PointerType::get(newType, *storageClass);

    rewriter.replaceOpWithNewOp<mlir::spirv::GlobalVariableOp>(
        op, ptrType, adaptor.getSymName());
    return mlir::success();
  }
};

class ConvertGetGlobalOp
    : public mlir::OpConversionPattern<mlir::memref::GetGlobalOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::GetGlobalOp op,
                  mlir::memref::GetGlobalOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto memrefType = mlir::dyn_cast<mlir::MemRefType>(op.getType());
    if (!memrefType)
      return mlir::failure();

    auto storageClass = convertStorageClass(memrefType.getMemorySpace());
    if (!storageClass)
      return mlir::failure();

    auto converter = getTypeConverter();
    assert(converter);
    auto resType = converter->convertType(memrefType);
    if (!resType)
      return mlir::failure();

    auto elemType = converter->convertType(memrefType.getElementType());
    if (!elemType)
      return mlir::failure();

    auto elemCount = memrefType.getNumElements();
    auto newType = mlir::spirv::ArrayType::get(elemType, elemCount);
    auto ptrType = mlir::spirv::PointerType::get(newType, *storageClass);

    auto loc = op->getLoc();
    mlir::Value res = rewriter.create<mlir::spirv::AddressOfOp>(
        loc, ptrType, adaptor.getName());
    if (res.getType() != resType)
      res = rewriter.create<mlir::spirv::BitcastOp>(loc, resType, res);

    rewriter.replaceOp(op, res);
    return mlir::success();
  }
};

// TODO: something better
class ConvertFunc : public mlir::OpConversionPattern<mlir::func::FuncOp> {
public:
  using mlir::OpConversionPattern<mlir::func::FuncOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::func::FuncOp op,
                  mlir::func::FuncOp::Adaptor /*adaptor*/,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (!op.getBody().empty())
      return mlir::failure();

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class ConvertAssert : public mlir::OpConversionPattern<mlir::cf::AssertOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::cf::AssertOp op, mlir::cf::AssertOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::spirv::KHRAssumeTrueOp>(op,
                                                              adaptor.getArg());
    return mlir::success();
  }
};

class ConvertPoison : public mlir::OpConversionPattern<mlir::ub::PoisonOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::ub::PoisonOp op, mlir::ub::PoisonOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto srcType = op.getType();
    if (!mlir::isa<mlir::MemRefType>(srcType))
      return mlir::failure();

    auto converter = getTypeConverter();
    auto type = converter->convertType(srcType);
    if (!type)
      return mlir::failure();

    rewriter.replaceOpWithNewOp<mlir::spirv::UndefOp>(op, type);
    return mlir::success();
  }
};

template <typename SourceOp, mlir::spirv::BuiltIn builtin>
class LaunchConfigConversion : public mlir::OpConversionPattern<SourceOp> {
public:
  using mlir::OpConversionPattern<SourceOp>::OpConversionPattern;

  LaunchConfigConversion(mlir::TypeConverter &converter,
                         mlir::MLIRContext *context)
      : mlir::OpConversionPattern<SourceOp>(converter, context,
                                            /*benefit*/ 10) {}

  mlir::LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Overrides original LaunchConfigConversion from spirv dialect.
    // Original LaunchConfigConversion replaces operations fro gpu dialect to
    // operations from spirv dialect and replaces index type to either int64 or
    // in32 depending on flag. This LaunchConfigConversion always assume result
    // of operation (WorkgroupId, etc) to be in64 but then converts to int32 if
    // needed

    auto *typeConverter =
        this->template getTypeConverter<mlir::SPIRVTypeConverter>();
    mlir::Type indexType = typeConverter->convertType(op.getType());

    if (not indexType)
      return rewriter.notifyMatchFailure(
          op, "Failed to find conversion for indexType");

    mlir::Type int64Type = rewriter.getIntegerType(64);
    mlir::Type int32Type = rewriter.getIntegerType(32);

    if (indexType != int32Type && indexType != int64Type)
      return rewriter.notifyMatchFailure(op, [&](mlir::Diagnostic &diag) {
        diag << "indexType should be converted either to int32 or to int64. "
                "But it is converted to "
             << indexType;
      });

    auto loc = op.getLoc();

    mlir::Value vector =
        mlir::spirv::getBuiltinVariableValue(op, builtin, int64Type, rewriter);
    mlir::Value dim = rewriter.create<mlir::spirv::CompositeExtractOp>(
        loc, int64Type, vector,
        rewriter.getI32ArrayAttr({static_cast<int32_t>(op.getDimension())}));

    if (int64Type != indexType) {
      auto maxIntAttr = rewriter.getIntegerAttr(
          int64Type,
          static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1);
      auto maxInt =
          rewriter.create<mlir::spirv::ConstantOp>(loc, int64Type, maxIntAttr);

      auto cmp = rewriter.create<mlir::spirv::SLessThanOp>(loc, dim, maxInt);
      rewriter.create<mlir::spirv::KHRAssumeTrueOp>(loc, cmp);

      mlir::Value cast =
          rewriter.create<mlir::spirv::SConvertOp>(loc, indexType, dim);
      rewriter.replaceOp(op, cast);
    } else {
      rewriter.replaceOp(op, dim);
    }

    return mlir::success();
  }
};

namespace {
using namespace mlir;

Region::iterator getBlockIt(Region &region, unsigned index) {
  return std::next(region.begin(), index);
}

template <typename OpTy>
class SCFToSPIRVPattern : public OpConversionPattern<OpTy> {
public:
  SCFToSPIRVPattern<OpTy>(MLIRContext *context, SPIRVTypeConverter &converter,
                          ScfToSPIRVContextImpl *scfToSPIRVContext)
      : OpConversionPattern<OpTy>::OpConversionPattern(converter, context,
                                                       /*benefit*/ 10),
        scfToSPIRVContext(scfToSPIRVContext), typeConverter(converter) {}

protected:
  ScfToSPIRVContextImpl *scfToSPIRVContext;
  // FIXME: We explicitly keep a reference of the type converter here instead of
  // passing it to OpConversionPattern during construction. This effectively
  // bypasses the conversion framework's automation on type conversion. This is
  // needed right now because the conversion framework will unconditionally
  // legalize all types used by SCF ops upon discovering them, for example, the
  // types of loop carried values. We use SPIR-V variables for those loop
  // carried values. Depending on the available capabilities, the SPIR-V
  // variable can be different, for example, cooperative matrix or normal
  // variable. We'd like to detach the conversion of the loop carried values
  // from the SCF ops (which is mainly a region). So we need to "mark" types
  // used by SCF ops as legal, if to use the conversion framework for type
  // conversion. There isn't a straightforward way to do that yet, as when
  // converting types, ops aren't taken into consideration. Therefore, we just
  // bypass the framework's type conversion for now.
  SPIRVTypeConverter &typeConverter;
};

// TODO: need fix upstream
struct WhileOpConversion final : SCFToSPIRVPattern<scf::WhileOp> {
  using SCFToSPIRVPattern::SCFToSPIRVPattern;

  LogicalResult
  matchAndRewrite(scf::WhileOp whileOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = whileOp.getLoc();
    auto loopOp = rewriter.create<spirv::LoopOp>(loc, spirv::LoopControl::None);
    loopOp.addEntryAndMergeBlock();

    OpBuilder::InsertionGuard guard(rewriter);

    Region &beforeRegion = whileOp.getBefore();
    Region &afterRegion = whileOp.getAfter();

    if (failed(rewriter.convertRegionTypes(&beforeRegion, typeConverter)) ||
        failed(rewriter.convertRegionTypes(&afterRegion, typeConverter)))
      return mlir::failure();

    Block &entryBlock = *loopOp.getEntryBlock();
    Block &beforeBlock = beforeRegion.front();
    Block &afterBlock = afterRegion.front();
    Block &mergeBlock = *loopOp.getMergeBlock();

    auto cond = cast<scf::ConditionOp>(beforeBlock.getTerminator());
    SmallVector<Value> condArgs;
    if (failed(rewriter.getRemappedValues(cond.getArgs(), condArgs)))
      return failure();

    Value conditionVal = rewriter.getRemappedValue(cond.getCondition());
    if (!conditionVal)
      return failure();

    auto yield = cast<scf::YieldOp>(afterBlock.getTerminator());
    SmallVector<Value> yieldArgs;
    if (failed(rewriter.getRemappedValues(yield.getResults(), yieldArgs)))
      return failure();

    // Move the while before block as the initial loop header block.
    rewriter.inlineRegionBefore(beforeRegion, loopOp.getBody(),
                                getBlockIt(loopOp.getBody(), 1));

    // Move the while after block as the initial loop body block.
    rewriter.inlineRegionBefore(afterRegion, loopOp.getBody(),
                                getBlockIt(loopOp.getBody(), 2));

    // Jump from the loop entry block to the loop header block.
    rewriter.setInsertionPointToEnd(&entryBlock);
    rewriter.create<spirv::BranchOp>(loc, &beforeBlock, adaptor.getInits());

    auto condLoc = cond.getLoc();

    SmallVector<Value> resultValues(condArgs.size());

    // For other SCF ops, the scf.yield op yields the value for the whole SCF
    // op. So we use the scf.yield op as the anchor to create/load/store SPIR-V
    // local variables. But for the scf.while op, the scf.yield op yields a
    // value for the before region, which may not matching the whole op's
    // result. Instead, the scf.condition op returns values matching the whole
    // op's results. So we need to create/load/store variables according to
    // that.
    for (const auto &it : llvm::enumerate(condArgs)) {
      auto res = it.value();
      auto i = it.index();
      auto pointerType =
          spirv::PointerType::get(res.getType(), spirv::StorageClass::Function);

      // Create local variables before the scf.while op.
      rewriter.setInsertionPoint(loopOp);
      auto alloc = rewriter.create<spirv::VariableOp>(
          condLoc, pointerType, spirv::StorageClass::Function,
          /*initializer=*/nullptr);

      // Load the final result values after the scf.while op.
      rewriter.setInsertionPointAfter(loopOp);
      auto loadResult = rewriter.create<spirv::LoadOp>(condLoc, alloc);
      resultValues[i] = loadResult;

      // Store the current iteration's result value.
      rewriter.setInsertionPointToEnd(&beforeBlock);
      rewriter.create<spirv::StoreOp>(condLoc, alloc, res);
    }

    rewriter.setInsertionPointToEnd(&beforeBlock);
    rewriter.replaceOpWithNewOp<spirv::BranchConditionalOp>(
        cond, conditionVal, &afterBlock, condArgs, &mergeBlock, std::nullopt);

    // Convert the scf.yield op to a branch back to the header block.
    rewriter.setInsertionPointToEnd(&afterBlock);
    rewriter.replaceOpWithNewOp<spirv::BranchOp>(yield, &beforeBlock,
                                                 yieldArgs);

    rewriter.replaceOp(whileOp, resultValues);
    return success();
  }
};
} // namespace

struct GPUToSpirvPass
    : public mlir::PassWrapper<GPUToSpirvPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GPUToSpirvPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::affine::AffineDialect>();
    registry.insert<mlir::spirv::SPIRVDialect>();
  }

  void runOnOperation() override {
    auto *context = &getContext();
    auto module = getOperation();

    llvm::SmallVector<mlir::Operation *, 1> kernelModules;
    mlir::OpBuilder builder(context);
    module.walk([&builder, &kernelModules](mlir::gpu::GPUModuleOp moduleOp) {
      // For each kernel module (should be only 1 for now, but that is not a
      // requirement here), clone the module for conversion because the
      // gpu.launch function still needs the kernel module.
      builder.setInsertionPoint(moduleOp.getOperation());
      kernelModules.push_back(builder.clone(*moduleOp.getOperation()));
    });

    for (auto kernelModule : kernelModules) {
      auto targetAttr = mlir::spirv::lookupTargetEnvOrDefault(kernelModule);
      auto target = mlir::SPIRVConversionTarget::get(targetAttr);

      auto use64bitIndexAttr = kernelModule->getAttrOfType<mlir::BoolAttr>(
          gpu_runtime::getUse64BitIndexAttrName());

      mlir::SPIRVConversionOptions options;
      options.use64bitIndex = true;

      auto use64bitIndexFlag =
          use64bitIndexAttr ? use64bitIndexAttr.getValue() : true;
      auto indexBits = use64bitIndexFlag ? 64 : 32;
      auto indexType = builder.getIntegerType(indexBits);

      mlir::SPIRVTypeConverter typeConverter(targetAttr, options);
      mlir::RewritePatternSet patterns(context);

      typeConverter.addConversion(
          [&typeConverter](mlir::MemRefType type) -> std::optional<mlir::Type> {
            auto srcElemType = type.getElementType();
            if (!srcElemType.isIntOrFloat() &&
                !mlir::isa<mlir::VectorType>(srcElemType))
              return mlir::Type(nullptr);

            auto elemType = typeConverter.convertType(srcElemType);
            if (!elemType)
              return mlir::Type(nullptr);

            auto sc =
                convertStorageClass(type.getMemorySpace(),
                                    mlir::spirv::StorageClass::CrossWorkgroup);

            return mlir::spirv::PointerType::get(elemType, sc);
          });

      // This conversion overrides spirv index conversion. options.use64bitIndex
      // not only determine index type but also affect Physical32/Physical64
      // aspect. We want to compile with Physical64 but have all benefits of
      // 32bit index
      typeConverter.addConversion(
          [indexType](mlir::IndexType type) -> std::optional<mlir::Type> {
            return indexType;
          });

      mlir::ScfToSPIRVContext scfToSpirvCtx;
      mlir::populateSCFToSPIRVPatterns(typeConverter, scfToSpirvCtx, patterns);
      mlir::populateGPUToSPIRVPatterns(typeConverter, patterns);
      mlir::populateFuncToSPIRVPatterns(typeConverter, patterns);
      mlir::cf::populateControlFlowToSPIRVPatterns(typeConverter, patterns);
      mlir::arith::populateArithToSPIRVPatterns(typeConverter, patterns);
      mlir::populateMathToSPIRVPatterns(typeConverter, patterns);
      mlir::populateMemRefToSPIRVPatterns(typeConverter, patterns);
      mlir::ub::populateUBToSPIRVConversionPatterns(typeConverter, patterns);

      patterns.insert<
          ConvertBitcastOp<numba::util::BitcastOp>,
          ConvertBitcastOp<numba::util::MemrefApplyOffsetOp>,
          ConvertBitcastOp<numba::util::MemrefBitcastOp>, ConvertAtomicRMW,
          ConvertI1SIndexCast, ConvertI1UIndexCast, AllocaOpPattern,
          ConvertFunc, ConvertAssert, ConvertBarrierOp, ConvertMemFenceOp,
          ConvertPoison, ConvertGlobalOp, ConvertGetGlobalOp,
          LaunchConfigConversion<mlir::gpu::BlockIdOp,
                                 mlir::spirv::BuiltIn::WorkgroupId>,
          LaunchConfigConversion<mlir::gpu::GridDimOp,
                                 mlir::spirv::BuiltIn::NumWorkgroups>,
          LaunchConfigConversion<mlir::gpu::BlockDimOp,
                                 mlir::spirv::BuiltIn::WorkgroupSize>,
          LaunchConfigConversion<mlir::gpu::ThreadIdOp,
                                 mlir::spirv::BuiltIn::LocalInvocationId>,
          LaunchConfigConversion<mlir::gpu::GlobalIdOp,
                                 mlir::spirv::BuiltIn::GlobalInvocationId>>(
          typeConverter, context);

      patterns.add<WhileOpConversion>(patterns.getContext(), typeConverter,
                                      scfToSpirvCtx.getImpl());

      if (failed(
              applyFullConversion(kernelModule, *target, std::move(patterns))))
        return signalPassFailure();
    }
  }
};

struct GpuIndexCastPass
    : public mlir::PassWrapper<GpuIndexCastPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GpuIndexCastPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::gpu::GPUDialect>();
  }

  void runOnOperation() override {
    auto *context = &getContext();
    auto module = getOperation();

    llvm::SmallVector<mlir::Value> kernelOperands;
    mlir::OpBuilder builder(context);
    module.walk([this, &context, &module, &builder,
                 &kernelOperands](mlir::gpu::LaunchFuncOp launchFuncOp) {
      // Converts kernel arguments of index type to int32 or int64

      auto gpuModule = module.lookupSymbol<mlir::gpu::GPUModuleOp>(
          launchFuncOp.getKernelModuleName());
      if (!gpuModule) {
        launchFuncOp.emitError("Failed to find GPUModuleOp with name ")
            << launchFuncOp.getKernelModuleName();
        return signalPassFailure();
      }

      builder.setInsertionPoint(launchFuncOp.getOperation());
      auto operands = launchFuncOp.getKernelOperands();
      auto loc = launchFuncOp.getLoc();

      auto use64bitIndexAttr = gpuModule->getAttrOfType<mlir::BoolAttr>(
          gpu_runtime::getUse64BitIndexAttrName());

      auto use64bitIndexFlag =
          use64bitIndexAttr ? use64bitIndexAttr.getValue() : true;
      auto indexBitSize = use64bitIndexFlag ? 64 : 32;
      kernelOperands.clear();
      kernelOperands.reserve(operands.size());
      for (auto &&operand : operands) {
        auto op_type = operand.getType();

        auto new_operand = operand;
        if (mlir::isa<mlir::IndexType>(op_type)) {
          new_operand = builder.create<mlir::arith::IndexCastOp>(
              loc, mlir::IntegerType::get(context, indexBitSize), operand);
        }

        kernelOperands.push_back(new_operand);
      }
      launchFuncOp.getKernelOperandsMutable().assign(kernelOperands);
    });
  }
};

template <typename Op, typename F>
static mlir::LogicalResult createGpuKernelLoad(mlir::PatternRewriter &builder,
                                               Op &&op, F &&func) {
  auto mod = op->template getParentOfType<mlir::ModuleOp>();
  if (!mod)
    return mlir::failure();

  auto gpuMod = mod.template lookupSymbol<mlir::gpu::GPUModuleOp>(
      op.getKernelModuleName());
  if (!gpuMod)
    return mlir::failure();

  auto gpuKernel =
      gpuMod.template lookupSymbol<mlir::gpu::GPUFuncOp>(op.getKernelName());
  if (!gpuKernel)
    return mlir::failure();

  auto queue = getGpuQueue(builder, op);
  if (!queue)
    return mlir::failure();

  auto loc = op.getLoc();
  auto module =
      builder.create<gpu_runtime::LoadGpuModuleOp>(loc, *queue, gpuMod);
  auto kernel =
      builder.create<gpu_runtime::GetGpuKernelOp>(loc, module, gpuKernel);
  auto newOp = func(builder, loc, *queue, kernel);
  builder.replaceOp(op, newOp.getResults());
  return mlir::success();
}

struct ExpandLaunchOp : public mlir::OpRewritePattern<mlir::gpu::LaunchFuncOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::gpu::LaunchFuncOp op,
                  mlir::PatternRewriter &rewriter) const override {
    return createGpuKernelLoad(
        rewriter, op,
        [&](mlir::OpBuilder &builder, mlir::Location loc, mlir::Value queue,
            mlir::Value kernel) {
          return builder.create<gpu_runtime::LaunchGpuKernelOp>(
              loc, queue, kernel, op.getGridSizeOperandValues(),
              op.getBlockSizeOperandValues(), op.getKernelOperands());
        });
  }
};

struct ExpandAllocOp : public mlir::OpRewritePattern<mlir::gpu::AllocOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::gpu::AllocOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto queue = getGpuQueue(rewriter, op);
    if (!queue)
      return mlir::failure();

    auto hostShared = op.getHostShared();
    mlir::Type token =
        op.getAsyncToken() ? op.getAsyncToken().getType() : nullptr;
    auto newOp = rewriter.create<gpu_runtime::GPUAllocOp>(
        op.getLoc(), op.getType(), token, op.getAsyncDependencies(), *queue,
        op.getDynamicSizes(), op.getSymbolOperands(), hostShared);

    newOp->setAttrs(op->getDiscardableAttrs());
    rewriter.replaceOp(op, newOp.getResults());
    return mlir::success();
  }
};

struct ExpandDeallocOp : public mlir::OpRewritePattern<mlir::gpu::DeallocOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::gpu::DeallocOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto queue = getGpuQueue(rewriter, op);
    if (!queue)
      return mlir::failure();

    rewriter.replaceOpWithNewOp<gpu_runtime::GPUDeallocOp>(
        op, op.getResultTypes(), op.getAsyncDependencies(), op.getMemref(),
        *queue);

    return mlir::success();
  }
};

struct ExpandSuggestBlockSizeOp
    : public mlir::OpRewritePattern<gpu_runtime::GPUSuggestBlockSizeOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(gpu_runtime::GPUSuggestBlockSizeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op.getKernel() || !op.getKernelRef())
      return mlir::failure();

    return createGpuKernelLoad(
        rewriter, op,
        [&](mlir::OpBuilder &builder, mlir::Location loc, mlir::Value queue,
            mlir::Value kernel) {
          return builder.create<gpu_runtime::GPUSuggestBlockSizeOp>(
              loc, queue, op.getGridSize(), kernel);
        });
  }
};

struct AbiAttrsPass
    : public mlir::PassWrapper<AbiAttrsPass,
                               mlir::OperationPass<mlir::gpu::GPUModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AbiAttrsPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::gpu::GPUDialect>();
    registry.insert<mlir::spirv::SPIRVDialect>();
  }

  void runOnOperation() override {
    auto gpuModule = getOperation();
    auto *context = &getContext();
    auto attrName =
        mlir::StringAttr::get(context, mlir::spirv::getEntryPointABIAttrName());
    auto abi = mlir::spirv::getEntryPointABIAttr(context);
    for (auto gpuFunc : gpuModule.getOps<mlir::gpu::GPUFuncOp>()) {
      if (!mlir::gpu::GPUDialect::isKernel(gpuFunc) ||
          gpuFunc->getAttr(attrName))
        continue;

      gpuFunc->setAttr(attrName, abi);
    }
  }
};

static mlir::spirv::TargetEnvAttr defaultCapsMapper(mlir::gpu::GPUModuleOp op) {
  auto context = op.getContext();
  namespace spirv = mlir::spirv;
  spirv::Capability caps[] = {
      // clang-format off
      spirv::Capability::Addresses,
      spirv::Capability::AtomicFloat32AddEXT,
      spirv::Capability::ExpectAssumeKHR,
      spirv::Capability::Float16,
      spirv::Capability::Float16Buffer,
      spirv::Capability::Float64,
      spirv::Capability::GenericPointer,
      spirv::Capability::Groups,
      spirv::Capability::Int16,
      spirv::Capability::Int64,
      spirv::Capability::Int8,
      spirv::Capability::Kernel,
      spirv::Capability::Linkage,
      spirv::Capability::Vector16,
      // clang-format on
  };
  spirv::Extension exts[] = {spirv::Extension::SPV_EXT_shader_atomic_float_add,
                             spirv::Extension::SPV_KHR_expect_assume};
  llvm::sort(caps);
  llvm::sort(exts);
  auto triple =
      spirv::VerCapExtAttr::get(spirv::Version::V_1_0, caps, exts, context);
  auto attr = spirv::TargetEnvAttr::get(
      triple, spirv::getDefaultResourceLimits(context),
      spirv::ClientAPI::OpenCL, spirv::Vendor::Unknown,
      spirv::DeviceType::Unknown, spirv::TargetEnvAttr::kUnknownDeviceID);
  return attr;
}

struct SetSPIRVCapabilitiesPass
    : public mlir::PassWrapper<SetSPIRVCapabilitiesPass,
                               mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SetSPIRVCapabilitiesPass)

  SetSPIRVCapabilitiesPass(
      std::function<mlir::spirv::TargetEnvAttr(mlir::gpu::GPUModuleOp)> m)
      : mapper(std::move(m)) {
    if (!mapper)
      mapper = &defaultCapsMapper;
  }

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::gpu::GPUDialect>();
    registry.insert<mlir::spirv::SPIRVDialect>();
  }

  void runOnOperation() override {
    assert(mapper && "Invalid mapper");
    namespace spirv = mlir::spirv;
    auto op = getOperation();
    op->walk([&](mlir::gpu::GPUModuleOp op) {
      if (auto attr = mapper(op))
        op->setAttr(spirv::getTargetEnvAttrName(), attr);
    });
  }

private:
  std::function<mlir::spirv::TargetEnvAttr(mlir::gpu::GPUModuleOp)> mapper;
};

struct SerializeSPIRVPass
    : public mlir::PassWrapper<SerializeSPIRVPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SerializeSPIRVPass)

  void runOnOperation() override {
    auto mod = getOperation();

    namespace gpu = mlir::gpu;
    namespace spirv = mlir::spirv;
    llvm::SmallVector<uint32_t, 0> spvBinary;
    for (auto gpuMod : mod.getOps<gpu::GPUModuleOp>()) {
      auto name = gpuMod.getName();
      auto isSameMod = [&](spirv::ModuleOp spvMod) -> bool {
        auto spvModName = spvMod.getName();
        return spvModName->consume_front("__spv__") && spvModName == name;
      };
      auto spvMods = mod.getOps<spirv::ModuleOp>();
      auto it = llvm::find_if(spvMods, isSameMod);
      if (it == spvMods.end()) {
        gpuMod.emitError() << "Unable to find corresponding SPIR-V module";
        signalPassFailure();
        return;
      }
      auto spvMod = *it;

      spvBinary.clear();
      if (mlir::failed(spirv::serialize(spvMod, spvBinary))) {
        spvMod.emitError() << "Failed to serialize SPIR-V module";
        signalPassFailure();
        return;
      }

      auto spvData =
          llvm::StringRef(reinterpret_cast<const char *>(spvBinary.data()),
                          spvBinary.size() * sizeof(uint32_t));
      auto spvAttr = mlir::StringAttr::get(&getContext(), spvData);
      gpuMod->setAttr(gpu::getDefaultGpuBinaryAnnotation(), spvAttr);
      spvMod->erase();
    }
  }
};

struct GPUExPass
    : public mlir::PassWrapper<GPUExPass, mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GPUExPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<gpu_runtime::GpuRuntimeDialect>();
    registry.insert<mlir::gpu::GPUDialect>();
  }

  void runOnOperation() override {
    auto *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);

    patterns.insert<ExpandLaunchOp, ExpandAllocOp, ExpandDeallocOp,
                    ExpandSuggestBlockSizeOp>(ctx);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                        std::move(patterns))))
      return signalPassFailure();
  }
};

struct ExpandDeviceFuncCallOp
    : public mlir::OpRewritePattern<mlir::func::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::func::CallOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!getGpuRegionEnv(op))
      return mlir::failure();

    auto mod = op->getParentOfType<mlir::ModuleOp>();
    if (!mod)
      return mlir::failure();

    auto funcName = op.getCallee();
    auto origFunc = mod.lookupSymbol<mlir::func::FuncOp>(funcName);
    if (!origFunc)
      return mlir::failure();

    auto deviceFuncAttr = origFunc->getAttrOfType<mlir::StringAttr>(
        gpu_runtime::getDeviceFuncAttrName());
    if (!deviceFuncAttr)
      return mlir::failure();

    auto queue = getGpuQueue(rewriter, op);
    if (!queue)
      return mlir::failure();

    auto deviceFunc = [&]() -> mlir::func::FuncOp {
      auto deviceFuncName = deviceFuncAttr.getValue();
      auto func = mod.lookupSymbol<mlir::func::FuncOp>(deviceFuncName);
      if (!func) {
        auto origFuncType = origFunc.getFunctionType();
        llvm::SmallVector<mlir::Type> newInputs;
        newInputs.emplace_back(queue->getType());
        auto inputs = origFuncType.getInputs();
        newInputs.append(inputs.begin(), inputs.end());

        auto funcType =
            rewriter.getFunctionType(newInputs, origFuncType.getResults());

        func = numba::addFunction(rewriter, mod, deviceFuncName, funcType);
        auto cifaceName = rewriter.getStringAttr("llvm.emit_c_interface");

        if (origFunc->hasAttr(cifaceName))
          func->setAttr(cifaceName, mlir::UnitAttr::get(rewriter.getContext()));
      }
      return func;
    }();

    auto oldArgs = op.getOperands();
    llvm::SmallVector<mlir::Value> newArgs;
    newArgs.emplace_back(*queue);
    newArgs.append(oldArgs.begin(), oldArgs.end());

    rewriter.replaceOpWithNewOp<mlir::func::CallOp>(op, deviceFunc, newArgs);
    return mlir::success();
  }
};

struct GenDeviceFuncsPass
    : public mlir::PassWrapper<GenDeviceFuncsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenDeviceFuncsPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::func::FuncDialect>();
  }

  void runOnOperation() override {
    auto *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);

    patterns.insert<ExpandDeviceFuncCallOp>(ctx);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                        std::move(patterns))))
      return signalPassFailure();
  }
};

static std::optional<mlir::TypedAttr> getNeutralValue(mlir::Block &block) {
  auto body = block.without_terminator();
  if (!llvm::hasSingleElement(body))
    return std::nullopt;

  return mlir::arith::getNeutralElement(&(*body.begin()));
}

static bool isInsideGPURegion(mlir::Operation *op) {
  assert(op && "Invalid op");
  return static_cast<bool>(getGpuRegionEnv(op));
}

struct TileParallelOp : public mlir::OpRewritePattern<mlir::scf::ParallelOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::ParallelOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // Process only loops inside gpu region.
    if (!isInsideGPURegion(op))
      return mlir::failure();

    // Process only outermost loops without mappings.
    if (op->getParentOfType<mlir::scf::ParallelOp>() ||
        op->hasAttr(mlir::gpu::getMappingAttrName()))
      return mlir::failure();

    auto reductionOps =
        llvm::to_vector(op.getBody()->getOps<mlir::scf::ReduceOp>());
    mlir::ValueRange initVals = op.getInitVals();
    assert(reductionOps.size() == initVals.size());

    llvm::SmallVector<mlir::TypedAttr> neutralValues;
    for (auto reduction : reductionOps) {
      auto neutralValue = getNeutralValue(reduction.getRegion().front());
      if (!neutralValue)
        return mlir::failure();

      neutralValues.emplace_back(*neutralValue);
    }

    auto oldLowerBounds = op.getLowerBound();
    auto oldUpperBounds = op.getUpperBound();
    auto oldSteps = op.getStep();
    auto oldLoopsCount = static_cast<unsigned>(oldSteps.size());

    const unsigned maxLoops = 3;
    // Only unit step is supported and iteration must start from 0.
    unsigned numLoops = 0;
    for (auto &&[start, step] : llvm::zip(oldLowerBounds.take_front(maxLoops),
                                          oldSteps.take_front(maxLoops)))
      if (mlir::isConstantIntValue(start, 0) &&
          mlir::isConstantIntValue(step, 1))
        ++numLoops;

    // No suitable loops.
    if (numLoops == 0)
      return mlir::failure();

    auto loc = op->getLoc();
    mlir::Value zero = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value one = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);

    std::array<mlir::Value, 3> globalSize;
    globalSize.fill(one);
    llvm::copy(oldUpperBounds.take_front(numLoops), globalSize.begin());

    std::optional<mlir::Value> queue;
    auto localSize =
        rewriter
            .create<gpu_runtime::GPUSuggestBlockSizeOp>(loc, queue, globalSize)
            ->getResults();

    llvm::SmallVector<mlir::Value> newLowerBounds;
    llvm::SmallVector<mlir::Value> newUpperBounds;
    llvm::SmallVector<mlir::Value> newSteps;

    // Insert grid vars.
    for (auto i : llvm::seq(0u, maxLoops)) {
      newLowerBounds.emplace_back(zero);
      newSteps.emplace_back(one);
      if (i < numLoops) {
        auto oldUpperBound = oldUpperBounds[i];
        mlir::Value newUpperBound = rewriter.create<mlir::arith::CeilDivUIOp>(
            loc, oldUpperBound, localSize[i]);
        newUpperBounds.emplace_back(newUpperBound);
      } else {
        newUpperBounds.emplace_back(one);
      }
    }

    // Insert block vars.
    for (auto i : llvm::seq(0u, maxLoops)) {
      newLowerBounds.emplace_back(zero);
      newSteps.emplace_back(one);
      if (i < numLoops) {
        newUpperBounds.emplace_back(localSize[i]);
      } else {
        newUpperBounds.emplace_back(one);
      }
    }

    for (auto i : llvm::seq(numLoops, oldLoopsCount)) {
      newLowerBounds.emplace_back(oldLowerBounds[i]);
      newUpperBounds.emplace_back(oldUpperBounds[i]);
      newSteps.emplace_back(oldSteps[i]);
    }

    auto newOp = rewriter.create<mlir::scf::ParallelOp>(
        loc, newLowerBounds, newUpperBounds, newSteps, initVals);
    mlir::Block *originalBlock = op.getBody();
    mlir::Block *newBlock = newOp.getBody();

    mlir::Value inBounds;
    llvm::SmallVector<mlir::Value> argMapping(oldLoopsCount);
    mlir::scf::IfOp ifOp;
    {
      mlir::OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(newBlock);
      for (auto i : llvm::seq(0u, oldLoopsCount)) {
        if (i < numLoops) {
          mlir::Value gridId = newBlock->getArgument(i);
          mlir::Value blockId = newBlock->getArgument(i + maxLoops);
          mlir::Value blockSize = localSize[i];
          mlir::Value gridSize = globalSize[i];
          mlir::Value val =
              rewriter.create<mlir::arith::MulIOp>(loc, gridId, blockSize);
          val = rewriter.create<mlir::arith::AddIOp>(loc, val, blockId);
          argMapping[i] = val;
          mlir::Value in = rewriter.createOrFold<mlir::arith::CmpIOp>(
              loc, mlir::arith::CmpIPredicate::slt, val, gridSize);
          if (0 == i) {
            inBounds = in;
          } else {
            inBounds =
                rewriter.createOrFold<mlir::arith::AndIOp>(loc, inBounds, in);
          }
        } else {
          argMapping[i] = newBlock->getArgument(i + maxLoops * 2 - numLoops);
        }
      }
      assert(inBounds);

      ifOp = [&]() -> mlir::scf::IfOp {
        if (!reductionOps.empty()) {
          llvm::SmallVector<mlir::Value> results;
          for (auto &&[i, val] : llvm::enumerate(initVals)) {
            auto constVal =
                rewriter.create<mlir::arith::ConstantOp>(loc, neutralValues[i]);
            results.emplace_back(constVal);
          }

          auto bodyBuilder = [&](mlir::OpBuilder &b, mlir::Location l) {
            b.create<mlir::scf::YieldOp>(l, results);
          };
          return rewriter.create<mlir::scf::IfOp>(loc, inBounds, bodyBuilder,
                                                  bodyBuilder);
        } else {
          auto thenBuilder = &mlir::scf::buildTerminatedBody;
          return rewriter.create<mlir::scf::IfOp>(loc, inBounds, thenBuilder);
        }
      }();

      newBlock = ifOp.thenBlock();
    }
    rewriter.eraseOp(newBlock->getTerminator()); // Erase exisitng yield.
    if (!reductionOps.empty()) {
      mlir::IRMapping mapper;
      mapper.map(originalBlock->getArguments(), argMapping);
      mlir::OpBuilder::InsertionGuard g(rewriter);
      auto loc = originalBlock->getTerminator()->getLoc();
      rewriter.eraseOp(originalBlock->getTerminator());
      rewriter.setInsertionPointToEnd(originalBlock);
      llvm::SmallVector<mlir::Value> results;
      for (auto &&[i, val] : llvm::enumerate(initVals)) {
        auto reductionArg =
            mapper.lookupOrDefault(reductionOps[i].getOperand());
        results.emplace_back(reductionArg);
      }
      rewriter.create<mlir::scf::YieldOp>(loc, results);

      rewriter.setInsertionPointAfter(ifOp);
      auto ifResults = ifOp.getResults();
      for (auto &&[i, reductionOp] : llvm::enumerate(reductionOps)) {
        mapper.map(reductionOp.getOperand(), ifResults[i]);
        rewriter.clone(*reductionOp, mapper);
        rewriter.eraseOp(reductionOp);
      }
    }
    rewriter.mergeBlocks(originalBlock, newBlock, argMapping);
    rewriter.replaceOp(op, newOp->getResults());

    auto newLoopsCount = static_cast<unsigned>(newSteps.size());
    auto identityMap = rewriter.getDimIdentityMap();
    llvm::SmallVector<mlir::gpu::ParallelLoopDimMappingAttr> mapping(
        newLoopsCount);
    for (auto i : llvm::seq(0u, newLoopsCount))
      mapping[i] = rewriter.getAttr<mlir::gpu::ParallelLoopDimMappingAttr>(
          getProcessor(i), identityMap, identityMap);

    return mlir::gpu::setMappingAttr(newOp, mapping);
  }
};

struct TileParallelLoopsForGPUPass
    : public mlir::PassWrapper<TileParallelLoopsForGPUPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TileParallelLoopsForGPUPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<gpu_runtime::GpuRuntimeDialect>();
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::gpu::GPUDialect>();
    registry.insert<mlir::scf::SCFDialect>();
    registry.insert<numba::util::NumbaUtilDialect>();
  }

  void runOnOperation() override {
    auto *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);

    patterns.insert<TileParallelOp>(ctx);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                        std::move(patterns))))
      return signalPassFailure();
  }
};

// Some manual fp conversion, denormals and nan/infs are not supported.
static mlir::Value f64Tof32(mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::Value src) {
  auto i64 = builder.getI64Type();
  auto srcI64 = builder.create<numba::util::BitcastOp>(loc, i64, src);

  auto zero = builder.create<mlir::arith::ConstantIntOp>(loc, 0, i64);
  auto absMask = builder.create<mlir::arith::ConstantIntOp>(
      loc, static_cast<int64_t>(0x7FFFFFFFFFFFFFFFULL), i64);

  mlir::Value absVal =
      builder.create<mlir::arith::AndIOp>(loc, srcI64, absMask);
  mlir::Value isZero = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::eq, absVal, zero);

  auto signShift = builder.create<mlir::arith::ConstantIntOp>(loc, 63, i64);
  auto expShift = builder.create<mlir::arith::ConstantIntOp>(loc, 52, i64);
  auto expMask = builder.create<mlir::arith::ConstantIntOp>(loc, 0x7FF, i64);
  auto manMask = builder.create<mlir::arith::ConstantIntOp>(
      loc, static_cast<int64_t>(0x000FFFFFFFFFFFFFULL), i64);
  auto b = builder.create<mlir::arith::ConstantIntOp>(loc, 1023 - 127, i64);
  auto _ff = builder.create<mlir::arith::ConstantIntOp>(loc, 0xFF, i64);
  auto _29 = builder.create<mlir::arith::ConstantIntOp>(loc, 29, i64);
  auto _23 = builder.create<mlir::arith::ConstantIntOp>(loc, 23, i64);
  auto _31 = builder.create<mlir::arith::ConstantIntOp>(loc, 31, i64);

  mlir::Value sign =
      builder.create<mlir::arith::ShRUIOp>(loc, srcI64, signShift);
  mlir::Value exponent =
      builder.create<mlir::arith::ShRUIOp>(loc, srcI64, expShift);
  exponent = builder.create<mlir::arith::AndIOp>(loc, exponent, expMask);
  mlir::Value mantissa =
      builder.create<mlir::arith::AndIOp>(loc, srcI64, manMask);
  exponent = builder.create<mlir::arith::SubIOp>(loc, exponent, b);

  exponent = builder.create<mlir::arith::AndIOp>(loc, exponent, _ff);
  mantissa = builder.create<mlir::arith::ShRUIOp>(loc, mantissa, _29);

  exponent = builder.create<mlir::arith::ShLIOp>(loc, exponent, _23);
  sign = builder.create<mlir::arith::ShLIOp>(loc, sign, _31);

  mlir::Value res = mantissa;
  res = builder.create<mlir::arith::OrIOp>(loc, res, exponent);
  res = builder.create<mlir::arith::OrIOp>(loc, res, sign);

  res = builder.create<mlir::arith::SelectOp>(loc, isZero, srcI64, res);

  res = builder.create<mlir::arith::TruncIOp>(loc, builder.getI32Type(), res);
  res = builder.create<mlir::arith::BitcastOp>(loc, builder.getF32Type(), res);
  return res;
}

// Some manual fp conversion, denormals and nan/infs are not supported.
static mlir::Value f32Tof64(mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::Value src, mlir::Type resType) {
  auto i32 = builder.getI32Type();
  mlir::Value srcI64 = builder.create<mlir::arith::BitcastOp>(loc, i32, src);

  auto i64 = builder.getI64Type();
  srcI64 = builder.create<mlir::arith::ExtUIOp>(loc, i64, srcI64);

  auto zero = builder.create<mlir::arith::ConstantIntOp>(loc, 0, i64);
  auto absMask = builder.create<mlir::arith::ConstantIntOp>(
      loc, static_cast<int64_t>(0x7FFFFFFFFFFFFFFFULL), i64);

  mlir::Value absVal =
      builder.create<mlir::arith::AndIOp>(loc, srcI64, absMask);
  mlir::Value isZero = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::eq, absVal, zero);

  auto signShift = builder.create<mlir::arith::ConstantIntOp>(loc, 31, i64);
  auto expShift = builder.create<mlir::arith::ConstantIntOp>(loc, 23, i64);
  auto expMask = builder.create<mlir::arith::ConstantIntOp>(loc, 0xFF, i64);
  auto manMask = builder.create<mlir::arith::ConstantIntOp>(loc, 0x7FFFFF, i64);
  auto b = builder.create<mlir::arith::ConstantIntOp>(loc, 1023 - 127, i64);
  auto _29 = builder.create<mlir::arith::ConstantIntOp>(loc, 29, i64);
  auto _52 = builder.create<mlir::arith::ConstantIntOp>(loc, 52, i64);
  auto _63 = builder.create<mlir::arith::ConstantIntOp>(loc, 63, i64);

  mlir::Value sign =
      builder.create<mlir::arith::ShRUIOp>(loc, srcI64, signShift);
  mlir::Value exponent =
      builder.create<mlir::arith::ShRUIOp>(loc, srcI64, expShift);
  exponent = builder.create<mlir::arith::AndIOp>(loc, exponent, expMask);
  mlir::Value mantissa =
      builder.create<mlir::arith::AndIOp>(loc, srcI64, manMask);

  mantissa = builder.create<mlir::arith::ShLIOp>(loc, mantissa, _29);
  exponent = builder.create<mlir::arith::AddIOp>(loc, exponent, b);

  exponent = builder.create<mlir::arith::ShLIOp>(loc, exponent, _52);
  sign = builder.create<mlir::arith::ShLIOp>(loc, sign, _63);

  mlir::Value res = mantissa;
  res = builder.create<mlir::arith::OrIOp>(loc, res, exponent);
  res = builder.create<mlir::arith::OrIOp>(loc, res, sign);

  res = builder.create<mlir::arith::SelectOp>(loc, isZero, srcI64, res);

  res = builder.create<numba::util::BitcastOp>(loc, resType, res);
  return res;
}

class ConvertF64LoadOp
    : public mlir::OpConversionPattern<mlir::memref::LoadOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::LoadOp op,
                  mlir::memref::LoadOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    assert(converter && "Invalid type converter");

    auto origResType = op.getType();
    if (!origResType.isF64())
      return mlir::success();

    auto resType = converter->convertType(origResType);
    if (!resType)
      return mlir::success();

    auto loc = op.getLoc();
    mlir::Value result = rewriter.create<mlir::memref::LoadOp>(
        loc, adaptor.getMemref(), adaptor.getIndices());
    result = f64Tof32(rewriter, loc, result);
    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

class ConvertF64StoreOp
    : public mlir::OpConversionPattern<mlir::memref::StoreOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::StoreOp op,
                  mlir::memref::StoreOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    assert(converter && "Invalid type converter");

    auto origType = op.getValue().getType();
    if (!origType.isF64())
      return mlir::failure();

    auto memref = adaptor.getMemref();
    auto memrefType = mlir::dyn_cast<mlir::MemRefType>(memref.getType());
    if (!memrefType)
      return mlir::failure();

    auto loc = op.getLoc();
    mlir::Value f64val = f32Tof64(rewriter, loc, adaptor.getValue(),
                                  memrefType.getElementType());
    rewriter.replaceOpWithNewOp<mlir::memref::StoreOp>(op, f64val, memref,
                                                       adaptor.getIndices());
    return mlir::success();
  }
};

class ConvertF64ReinterpretCastOp
    : public mlir::OpConversionPattern<mlir::memref::ReinterpretCastOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::ReinterpretCastOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    assert(converter && "Invalid type converter");

    auto resType = mlir::dyn_cast_or_null<mlir::MemRefType>(
        converter->convertType(op.getType()));
    if (!resType)
      return mlir::failure();

    auto offsets = mlir::getMixedValues(adaptor.getStaticOffsets(),
                                        adaptor.getOffsets(), rewriter);
    auto sizes = mlir::getMixedValues(adaptor.getStaticSizes(),
                                      adaptor.getSizes(), rewriter);
    auto strides = mlir::getMixedValues(adaptor.getStaticStrides(),
                                        adaptor.getStrides(), rewriter);

    rewriter.replaceOpWithNewOp<mlir::memref::ReinterpretCastOp>(
        op, resType, adaptor.getSource(), offsets.front(), sizes, strides);
    return mlir::success();
  }
};

class ConvertF64ApplyOffset
    : public mlir::OpConversionPattern<numba::util::MemrefApplyOffsetOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(numba::util::MemrefApplyOffsetOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    assert(converter && "Invalid type converter");

    auto resType = converter->convertType(op.getType());
    if (!resType)
      return rewriter.notifyMatchFailure(op, "Failed to convert result type");

    rewriter.replaceOpWithNewOp<numba::util::MemrefApplyOffsetOp>(
        op, resType, adaptor.getSource());
    return mlir::success();
  }
};

struct TruncateF64ForGPUPass
    : public mlir::PassWrapper<TruncateF64ForGPUPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TruncateF64ForGPUPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::gpu::GPUDialect>();
    registry.insert<mlir::math::MathDialect>();
    registry.insert<mlir::memref::MemRefDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();

    auto *ctx = &getContext();
    mlir::ConversionTarget target(*ctx);
    mlir::TypeConverter converter;

    // Convert unknown types to itself
    converter.addConversion([](mlir::Type type) { return type; });

    converter.addConversion([](mlir::Float64Type type) {
      return mlir::Float32Type::get(type.getContext());
    });

    converter.addConversion(
        [](mlir::MemRefType type) -> std::optional<mlir::Type> {
          if (!type.getElementType().isF64())
            return std::nullopt;

          int64_t shape[] = {2};
          auto elemType = mlir::IntegerType::get(type.getContext(), 32);
          auto newType = mlir::VectorType::get(shape, elemType);
          return type.clone(newType);
        });

    auto addCast = [](mlir::OpBuilder &builder, mlir::Type dstType,
                      mlir::ValueRange inputs,
                      mlir::Location loc) -> std::optional<mlir::Value> {
      if (inputs.size() != 1)
        return std::nullopt;

      auto src = inputs.front();
      auto srcType = src.getType();
      if (srcType.isF32() && dstType.isF64())
        return builder.create<mlir::arith::ExtFOp>(loc, dstType, src)
            .getResult();

      if (srcType.isF64() && dstType.isF32())
        return builder.create<mlir::arith::TruncFOp>(loc, dstType, src)
            .getResult();

      if (mlir::isa<mlir::MemRefType>(srcType) &&
          mlir::isa<mlir::MemRefType>(dstType))
        return builder.create<numba::util::MemrefBitcastOp>(loc, dstType, src)
            .getResult();

      return std::nullopt;
    };
    converter.addArgumentMaterialization(addCast);
    converter.addSourceMaterialization(addCast);
    converter.addTargetMaterialization(addCast);

    mlir::RewritePatternSet patterns(ctx);

    numba::populateArithConversionRewritesAndTarget(converter, patterns,
                                                    target);
    numba::populateMathConversionRewritesAndTarget(converter, patterns, target);
    numba::populateControlFlowTypeConversionRewritesAndTarget(converter,
                                                              patterns, target);
    numba::populateTupleTypeConversionRewritesAndTarget(converter, patterns,
                                                        target);

    patterns.insert<ConvertF64LoadOp, ConvertF64StoreOp,
                    ConvertF64ReinterpretCastOp, ConvertF64ApplyOffset>(
        converter, ctx);

    target.addDynamicallyLegalOp<mlir::memref::LoadOp, mlir::memref::StoreOp,
                                 mlir::memref::ReinterpretCastOp,
                                 numba::util::MemrefApplyOffsetOp>(
        [&converter](mlir::Operation *op) -> std::optional<bool> {
          if (converter.isLegal(op))
            return true;

          return std::nullopt;
        });

    mlir::FrozenRewritePatternSet frozenPatterns(std::move(patterns));

    llvm::SmallVector<mlir::Value> newArgs;
    mlir::OpBuilder builder(ctx);
    for (auto gpuModule : module.getOps<mlir::gpu::GPUModuleOp>()) {
      auto truncAttr = gpuModule->getAttrOfType<mlir::BoolAttr>(
          gpu_runtime::getFp64TruncateAttrName());
      if (truncAttr && !truncAttr.getValue())
        continue;

      auto targetEnv = mlir::spirv::lookupTargetEnv(gpuModule);
      if (!targetEnv) {
        gpuModule->emitError("TargetEnv not found");
        return signalPassFailure();
      }

      if (!truncAttr || !truncAttr.getValue()) {
        auto caps = targetEnv.getCapabilities();
        if (llvm::is_contained(caps, mlir::spirv::Capability::Float64))
          continue;
      }

      for (auto gpuFunc : gpuModule.getOps<mlir::gpu::GPUFuncOp>()) {
        auto origSig = gpuFunc.getFunctionType();
        if (mlir::failed(
                mlir::applyPartialConversion(gpuFunc, target, frozenPatterns)))
          return signalPassFailure();

        auto newSig = gpuFunc.getFunctionType();
        if (origSig == newSig)
          continue;

        auto funcUses = mlir::SymbolTable::getSymbolUses(gpuFunc, module);
        if (!funcUses)
          continue;

        for (auto use : llvm::make_early_inc_range(*funcUses)) {
          auto user = use.getUser();
          if (mlir::isa<gpu_runtime::GPUSuggestBlockSizeOp>(user))
            continue;

          auto launch = mlir::dyn_cast<mlir::gpu::LaunchFuncOp>(user);
          if (!launch) {
            user->emitError("Unknown gpu func user");
            return signalPassFailure();
          }

          builder.setInsertionPoint(launch);

          newArgs.clear();
          newArgs.reserve(launch.getNumKernelOperands());
          for (auto &&[origArg, newType] :
               llvm::zip(launch.getKernelOperands(), newSig.getInputs())) {
            auto origType = origArg.getType();
            if (newType == origType) {
              newArgs.emplace_back(origArg);
            } else if (origType.isF64() && newType.isF32()) {
              auto loc = launch.getLoc();
              mlir::Value newVal =
                  builder.create<mlir::arith::TruncFOp>(loc, newType, origArg);
              newArgs.emplace_back(newVal);
            } else if (mlir::isa<mlir::MemRefType>(origType) &&
                       mlir::isa<mlir::MemRefType>(newType)) {
              auto loc = launch.getLoc();
              mlir::Value newVal = builder.create<numba::util::MemrefBitcastOp>(
                  loc, newType, origArg);
              newArgs.emplace_back(newVal);
            } else {
              launch->emitError("Incompatible types: ")
                  << origType << " and " << newType;
              return signalPassFailure();
            }
          }

          launch.getKernelOperandsMutable().assign(newArgs);
        }
      }
    }
  }
};

struct InsertGPUGlobalReduce
    : public mlir::OpRewritePattern<mlir::scf::ParallelOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::ParallelOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // Process only outermost loops with mappings.
    if (op->getParentOfType<mlir::scf::ParallelOp>() ||
        !op->hasAttr(mlir::gpu::getMappingAttrName()))
      return mlir::failure();

    // Check if there any reductions.
    if (op.getInitVals().empty())
      return mlir::failure();

    auto reductionOps = op.getBody()->getOps<mlir::scf::ReduceOp>();
    assert(static_cast<size_t>(
               std::distance(reductionOps.begin(), reductionOps.end())) ==
           op.getInitVals().size());

    llvm::SmallVector<mlir::scf::ReduceOp> reductionOpsVec(reductionOps.begin(),
                                                           reductionOps.end());

    llvm::SmallVector<mlir::Value> results;
    results.reserve(op.getInitVals().size());

    auto loc = op.getLoc();
    mlir::IRMapping mapper;
    mlir::OpBuilder::InsertionGuard g(rewriter);

    auto loopBlock = op.getBody();
    rewriter.setInsertionPointToStart(loopBlock);
    mlir::Value cond;
    for (auto &&[lb, arg] :
         llvm::zip(op.getLowerBound(), loopBlock->getArguments())) {
      mlir::Value eq = rewriter.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::eq, arg, lb);
      if (!cond) {
        cond = eq;
      } else {
        cond = rewriter.create<mlir::arith::AndIOp>(loc, cond, eq);
      }
    }

    for (auto &&[reduce, init] : llvm::zip(reductionOpsVec, op.getInitVals())) {
      auto reduceType = init.getType();
      auto memrefType = mlir::MemRefType::get(std::nullopt, reduceType);

      rewriter.setInsertionPoint(op);
      mlir::Value array =
          rewriter
              .create<mlir::gpu::AllocOp>(
                  loc, memrefType, /*asyncToken*/ nullptr,
                  /*asyncDeps*/ std::nullopt, /*dynSizes*/ std::nullopt,
                  /*symbols*/ std::nullopt, /*hostShared*/ true)
              .getMemref();

      auto &reduceRegion = reduce.getReductionOperator();

      rewriter.setInsertionPointAfter(op);
      mlir::Value res = rewriter.create<mlir::memref::LoadOp>(loc, array);
      rewriter.create<mlir::gpu::DeallocOp>(loc, /*asyncToken*/ mlir::Type(),
                                            /*asyncDeps*/ std::nullopt, array);

      auto &reduceBlock = reduceRegion.front();
      mapper.clear();
      mapper.map(reduceBlock.getArgument(0), res);
      mapper.map(reduceBlock.getArgument(1), init);
      for (auto &innerOp : reduceBlock.without_terminator())
        rewriter.clone(innerOp, mapper);

      auto term =
          mlir::cast<mlir::scf::ReduceReturnOp>(reduceBlock.getTerminator());
      auto termResult = term.getResult();
      results.emplace_back(mapper.lookupOrNull(termResult));
      assert(results.back());

      rewriter.setInsertionPoint(reduce);
      auto newReduce = rewriter.create<gpu_runtime::GPUGlobalReduceOp>(
          reduce.getLoc(), reduce.getOperand(), array);

      auto &newRegion = newReduce.getRegion();
      rewriter.inlineRegionBefore(reduceRegion, newRegion, newRegion.end());

      rewriter.setInsertionPoint(term);
      rewriter.create<gpu_runtime::GPUGlobalReduceYieldOp>(term.getLoc(),
                                                           termResult);

      rewriter.eraseOp(term);
      rewriter.eraseOp(reduce);
    }

    rewriter.setInsertionPoint(op);
    auto newParallel = rewriter.create<mlir::scf::ParallelOp>(
        loc, op.getLowerBound(), op.getUpperBound(), op.getStep());
    auto &newParallelRegion = newParallel.getRegion();
    rewriter.eraseBlock(&newParallelRegion.front());
    rewriter.inlineRegionBefore(op.getRegion(), newParallelRegion,
                                newParallelRegion.end());

    rewriter.replaceOp(op, results);
    return mlir::success();
  }
};

struct InsertGPUGlobalReducePass
    : public mlir::PassWrapper<InsertGPUGlobalReducePass,
                               mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InsertGPUGlobalReducePass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<gpu_runtime::GpuRuntimeDialect>();
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<mlir::scf::SCFDialect>();
  }

  void runOnOperation() override {
    auto *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);

    patterns.insert<InsertGPUGlobalReduce>(ctx);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                        std::move(patterns))))
      return signalPassFailure();
  }
};

static mlir::Value computeGPUDimsProd(mlir::OpBuilder &builder,
                                      mlir::Location loc, mlir::Value x,
                                      mlir::Value y, mlir::Value z) {
  mlir::Value tmp = builder.create<mlir::arith::MulIOp>(loc, x, y);
  return builder.create<mlir::arith::MulIOp>(loc, tmp, z);
}

static mlir::Value isZeroIds(mlir::OpBuilder &builder, mlir::Location loc,
                             const mlir::gpu::KernelDim3 &ids) {
  mlir::Value zero = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
  mlir::Value eq = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::eq, ids.x, zero);
  mlir::Value tmp = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::eq, ids.y, zero);
  tmp = builder.create<mlir::arith::AndIOp>(loc, tmp, eq);
  eq = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq,
                                           ids.z, zero);
  return builder.create<mlir::arith::AndIOp>(loc, tmp, eq);
}

static mlir::Value computeLinearBlockId(mlir::OpBuilder &builder,
                                        mlir::Location loc,
                                        const mlir::gpu::KernelDim3 &gridSizes,
                                        const mlir::gpu::KernelDim3 &blockIds) {
  mlir::Value tmp =
      builder.create<mlir::arith::MulIOp>(loc, gridSizes.x, blockIds.y);
  mlir::Value ret = builder.create<mlir::arith::AddIOp>(loc, blockIds.x, tmp);
  tmp = builder.create<mlir::arith::MulIOp>(loc, gridSizes.x, gridSizes.y);
  tmp = builder.create<mlir::arith::MulIOp>(loc, tmp, blockIds.z);
  return builder.create<mlir::arith::AddIOp>(loc, ret, tmp);
}

struct LowerGPUGlobalReduce
    : public mlir::OpRewritePattern<gpu_runtime::GPUGlobalReduceOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(gpu_runtime::GPUGlobalReduceOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // Launch op must be a direct parent.
    auto launch = mlir::dyn_cast<mlir::gpu::LaunchOp>(op->getParentOp());
    if (!launch)
      return mlir::failure();

    auto initAttr = getNeutralValue(op.getRegion().front());
    if (!initAttr)
      return mlir::failure();

    auto launchLoc = launch.getLoc();
    mlir::OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(launch);

    mlir::Value resultArray = op.getTarget();

    mlir::Value numWorkGroupsExternal =
        computeGPUDimsProd(rewriter, launchLoc, launch.getGridSizeX(),
                           launch.getGridSizeY(), launch.getGridSizeZ());

    const int64_t shape[] = {mlir::ShapedType::kDynamic};
    auto arrayType = mlir::MemRefType::get(shape, op.getValue().getType());
    mlir::Value reduceArray =
        rewriter
            .create<mlir::gpu::AllocOp>(
                launchLoc, arrayType, /*asyncToken*/ nullptr,
                /*asyncDeps*/ std::nullopt, numWorkGroupsExternal,
                /*symbols*/ std::nullopt, /*hostShared*/ true)
            .getMemref();

    auto loc = op.getLoc();

    rewriter.setInsertionPoint(op);
    auto allReduce = rewriter.create<mlir::gpu::AllReduceOp>(
        loc, op.getValue(), mlir::gpu::AllReduceOperationAttr{},
        /*uniform*/ true);
    auto &newRegion = allReduce.getRegion();
    rewriter.inlineRegionBefore(op.getRegion(), newRegion, newRegion.end());

    auto &reduceBlock = newRegion.front();
    {
      mlir::OpBuilder::InsertionGuard g1(rewriter);
      auto term = mlir::cast<gpu_runtime::GPUGlobalReduceYieldOp>(
          reduceBlock.getTerminator());
      rewriter.setInsertionPoint(term);
      rewriter.replaceOpWithNewOp<mlir::gpu::YieldOp>(term, term.getResult());
    }

    mlir::gpu::KernelDim3 threadIds = launch.getThreadIds();
    mlir::gpu::KernelDim3 blockIds = launch.getBlockIds();
    mlir::gpu::KernelDim3 gridSizes = launch.getGridSize();

    mlir::Value linearBlockId =
        computeLinearBlockId(rewriter, loc, gridSizes, blockIds);

    mlir::Value isZeroThread = isZeroIds(rewriter, loc, threadIds);

    auto condWriteBuilder = [&](mlir::OpBuilder &b, mlir::Location l) {
      mlir::Value result = allReduce.getResult();
      b.create<mlir::memref::StoreOp>(l, result, reduceArray, linearBlockId);
      b.create<mlir::scf::YieldOp>(l);
    };

    rewriter.create<mlir::scf::IfOp>(loc, isZeroThread, condWriteBuilder);

    rewriter.setInsertionPointAfter(launch);
    mlir::Value zero =
        rewriter.create<mlir::arith::ConstantIndexOp>(launchLoc, 0);
    mlir::Value one =
        rewriter.create<mlir::arith::ConstantIndexOp>(launchLoc, 1);

    auto finalReduceBodyBuilder = [&](mlir::OpBuilder &b, mlir::Location l,
                                      mlir::ValueRange iters,
                                      mlir::ValueRange) {
      assert(iters.size() == 1);
      mlir::Value value =
          b.create<mlir::memref::LoadOp>(l, reduceArray, iters.front());
      auto reduce = b.create<mlir::scf::ReduceOp>(l, value);
      auto &finalReduceBlock = reduce.getRegion().front();

      mlir::IRMapping mapper;
      mapper.map(reduceBlock.getArguments(), finalReduceBlock.getArguments());

      {
        mlir::OpBuilder::InsertionGuard g1(b);
        b.setInsertionPointToStart(&finalReduceBlock);
        for (auto &op : reduceBlock.without_terminator())
          b.clone(op, mapper);

        auto term = mlir::cast<mlir::gpu::YieldOp>(reduceBlock.getTerminator());
        auto result = mapper.lookupOrDefault(term.getValues().front());
        b.create<mlir::scf::ReduceReturnOp>(l, result);
      }
      b.create<mlir::scf::YieldOp>(l);
    };

    mlir::Value initVal = rewriter.create<mlir::arith::ConstantOp>(
        launchLoc, mlir::cast<mlir::TypedAttr>(*initAttr));
    auto loopOp = rewriter.create<mlir::scf::ParallelOp>(
        launchLoc, zero, numWorkGroupsExternal, one, initVal,
        finalReduceBodyBuilder);

    rewriter.create<mlir::memref::StoreOp>(launchLoc, loopOp->getResult(0),
                                           resultArray);

    rewriter.create<mlir::gpu::DeallocOp>(
        launchLoc, /*asyncToken*/ mlir::Type(), /*asyncDeps*/ std::nullopt,
        reduceArray);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

template <typename Op, mlir::gpu::AllReduceOperation ReduceOp>
static std::optional<mlir::gpu::AllReduceOperation>
convertAllReduceOp(mlir::Operation *op) {
  if (mlir::isa<Op>(op))
    return ReduceOp;

  return std::nullopt;
}

struct AllReduceRemoveRegion
    : public mlir::OpRewritePattern<mlir::gpu::AllReduceOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::gpu::AllReduceOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto &region = op.getBody();
    if (region.empty())
      return mlir::failure();

    auto &block = region.front();
    auto ops = block.without_terminator();
    if (!llvm::hasSingleElement(ops))
      return mlir::failure();

    using RedOp = mlir::gpu::AllReduceOperation;
    using Handler = std::optional<RedOp> (*)(mlir::Operation *);
    const Handler handlers[] = {
        &convertAllReduceOp<mlir::arith::AddIOp, RedOp::ADD>,
        &convertAllReduceOp<mlir::arith::AddFOp, RedOp::ADD>,
        &convertAllReduceOp<mlir::arith::AndIOp, RedOp::AND>,
        &convertAllReduceOp<mlir::arith::XOrIOp, RedOp::XOR>,
        &convertAllReduceOp<mlir::arith::OrIOp, RedOp::OR>,
        &convertAllReduceOp<mlir::arith::MulIOp, RedOp::MUL>,
        &convertAllReduceOp<mlir::arith::MulFOp, RedOp::MUL>,
        &convertAllReduceOp<mlir::arith::MaxSIOp, RedOp::MAX>,
        &convertAllReduceOp<mlir::arith::MaximumFOp, RedOp::MAX>,
        &convertAllReduceOp<mlir::arith::MinSIOp, RedOp::MIN>,
        &convertAllReduceOp<mlir::arith::MinimumFOp, RedOp::MIN>,
    };

    auto result = [&]() -> std::optional<RedOp> {
      auto &reduceOp = *ops.begin();
      for (auto h : handlers)
        if (auto res = h(&reduceOp))
          return *res;

      return std::nullopt;
    }();
    if (!result)
      return mlir::failure();

    auto attr = mlir::gpu::AllReduceOperationAttr::get(getContext(), *result);
    rewriter.replaceOpWithNewOp<mlir::gpu::AllReduceOp>(op, op.getValue(), attr,
                                                        op.getUniform());

    return mlir::success();
  }
};

struct LowerGPUGlobalReducePass
    : public mlir::PassWrapper<LowerGPUGlobalReducePass,
                               mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerGPUGlobalReducePass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<gpu_runtime::GpuRuntimeDialect>();
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::gpu::GPUDialect>();
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<mlir::scf::SCFDialect>();
  }

  void runOnOperation() override {
    auto *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);

    patterns.insert<LowerGPUGlobalReduce, AllReduceRemoveRegion>(ctx);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                        std::move(patterns))))
      return signalPassFailure();
  }
};

/// The general idea of this transform is to assign weight to each scf.parallel
/// index arg and sort idices according to this weight.
struct SortSCFParallel : public mlir::OpRewritePattern<mlir::scf::ParallelOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::ParallelOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getParentOfType<mlir::scf::ParallelOp>())
      return rewriter.notifyMatchFailure(op, "must be a top-level parallel op");

    if (!isInsideGPURegion(op))
      return rewriter.notifyMatchFailure(op, "must be inside GPU region");

    using Arg = std::pair<unsigned, int>;

    llvm::SmallVector<Arg> args(op.getNumLoops());
    mlir::ValueRange indVars = op.getInductionVars();
    for (auto &&[i, arg] : llvm::enumerate(indVars))
      args[i] = std::pair(static_cast<unsigned>(i), 0);

    auto addWeight = [&](mlir::Value idx, int weight) {
      for (auto &&[argi, w] : args) {
        auto var = indVars[argi];
        if (var == idx) {
          w += weight;
          return;
        }
      }
    };

    auto visitor = [&](mlir::Operation *bodyOp) {
      mlir::Value memref;
      mlir::ValueRange indices;
      if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(bodyOp)) {
        memref = store.getMemRef();
        indices = store.getIndices();
      } else if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(bodyOp)) {
        memref = load.getMemRef();
        indices = load.getIndices();
      } else {
        return;
      }

      // Only process identity (c-contigious) memrefs for now.
      if (!mlir::cast<mlir::MemRefType>(memref.getType())
               .getLayout()
               .isIdentity())
        return;

      // Assign more weight to the rightmost indices, so linear accesses more
      // likely to be mapped to loop 0 (which is mapped to get_global_id(0)).
      auto numDims = static_cast<int>(indices.size());
      for (auto i : llvm::seq(0, numDims)) {
        auto weight = (i + 1) * 10 - (numDims - 1) * 10;
        addWeight(indices[static_cast<unsigned>(i)], weight);
      }
    };
    op.walk(visitor);

    std::stable_sort(args.begin(), args.end(),
                     [](auto &a, auto &b) { return a.second > b.second; });

    auto isSame = [&]() -> bool {
      for (auto &&[i, arg] : llvm::enumerate(indVars))
        if (indVars[args[i].first] != arg)
          return false;

      return true;
    }();

    if (isSame)
      return mlir::failure();

    auto numVars = static_cast<unsigned>(indVars.size());
    llvm::SmallVector<mlir::Value> newLowerBounds(numVars);
    llvm::SmallVector<mlir::Value> newUpperBounds(numVars);
    llvm::SmallVector<mlir::Value> newSteps(numVars);
    for (auto i : llvm::seq(0u, numVars)) {
      auto m = args[i].first;
      newLowerBounds[i] = op.getLowerBound()[m];
      newUpperBounds[i] = op.getUpperBound()[m];
      newSteps[i] = op.getStep()[m];
    }

    auto loc = op.getLoc();
    auto newOp = rewriter.create<mlir::scf::ParallelOp>(
        loc, newLowerBounds, newUpperBounds, newSteps, op.getInitVals());
    auto newBody = newOp.getBody();
    rewriter.eraseOp(newBody->getTerminator());

    llvm::SmallVector<mlir::Value> indVarMapped(numVars);
    for (auto i : llvm::seq(0u, numVars)) {
      auto m = args[i].first;
      indVarMapped[m] = newOp.getInductionVars()[i];
    }

    auto oldBody = op.getBody();
    rewriter.mergeBlocks(oldBody, newBody, indVarMapped);
    rewriter.replaceOp(op, newOp.getResults());
    return mlir::success();
  }
};

struct SortParallelLoosForGPU
    : public mlir::PassWrapper<SortParallelLoosForGPU,
                               mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SortParallelLoosForGPU)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<mlir::scf::SCFDialect>();
  }

  void runOnOperation() override {
    auto *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);

    patterns.insert<SortSCFParallel>(ctx);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                        std::move(patterns))))
      return signalPassFailure();
  }
};

static mlir::gpu::AllocOp convertToGPUAloc(mlir::OpBuilder &builder,
                                           mlir::memref::AllocOp op,
                                           numba::GpuAllocType allocType) {
  bool hostShared = (allocType == numba::GpuAllocType::Shared);
  auto type = op.getType();
  auto ret = builder.create<mlir::gpu::AllocOp>(
      op.getLoc(), type, /*asyncToken*/ nullptr, /*asyncDeps*/ std::nullopt,
      op.getDynamicSizes(), op.getSymbolOperands(), hostShared);
  mlir::ValueRange results(ret.getMemref());
  op.getResult().replaceAllUsesWith(ret.getMemref());
  op->erase();
  if (allocType == numba::GpuAllocType::Host)
    ret->setAttr(gpu_runtime::getHostAllocAttrName(), builder.getUnitAttr());

  return ret;
}

struct CreateGPUAllocPass
    : public mlir::PassWrapper<CreateGPUAllocPass, mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CreateGPUAllocPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::gpu::GPUDialect>();
    registry.insert<mlir::memref::MemRefDialect>();
  }

  void runOnOperation() override {
    llvm::SmallVector<mlir::memref::AllocOp> allocs;
    getOperation()->walk([&](mlir::memref::AllocOp alloc) {
      if (!isInsideGPURegion(alloc))
        return;

      allocs.emplace_back(alloc);
    });

    if (allocs.empty())
      return markAllAnalysesPreserved();

    mlir::OpBuilder builder(&getContext());
    auto deviceAttr = builder.getStringAttr("device");
    auto sharedAttr = builder.getStringAttr("shared");
    auto hostAttr = builder.getStringAttr("host");
    for (auto &&alloc : allocs) {
      auto env = getGpuRegionEnv(alloc);
      assert(env);

      builder.setInsertionPoint(alloc);
      auto usmType = env.getUsmType();
      if (usmType == deviceAttr) {
        convertToGPUAloc(builder, alloc, numba::GpuAllocType::Device);
      } else if (usmType == sharedAttr) {
        convertToGPUAloc(builder, alloc, numba::GpuAllocType::Shared);
      } else if (usmType == hostAttr) {
        convertToGPUAloc(builder, alloc, numba::GpuAllocType::Host);
      } else {
        alloc->emitError("Unknown usm_type value: ") << usmType;
        return signalPassFailure();
      }
    }
  }
};
} // namespace

// Expose the passes to the outside world
std::unique_ptr<mlir::Pass> gpu_runtime::createAbiAttrsPass() {
  return std::make_unique<AbiAttrsPass>();
}

std::unique_ptr<mlir::Pass> gpu_runtime::createSetSPIRVCapabilitiesPass(
    std::function<mlir::spirv::TargetEnvAttr(mlir::gpu::GPUModuleOp)> mapper) {
  return std::make_unique<SetSPIRVCapabilitiesPass>(std::move(mapper));
}

std::unique_ptr<mlir::Pass> gpu_runtime::createGPUToSpirvPass() {
  return std::make_unique<GPUToSpirvPass>();
}

std::unique_ptr<mlir::Pass> gpu_runtime::createGpuIndexCastPass() {
  return std::make_unique<GpuIndexCastPass>();
}

std::unique_ptr<mlir::Pass> gpu_runtime::createInsertGPUAllocsPass() {
  return std::make_unique<InsertGPUAllocs>();
}

std::unique_ptr<mlir::Pass> gpu_runtime::createConvertGPUDeallocsPass() {
  return std::make_unique<ConvertGPUDeallocsPass>();
}

std::unique_ptr<mlir::Pass> gpu_runtime::createSerializeSPIRVPass() {
  return std::make_unique<SerializeSPIRVPass>();
}

std::unique_ptr<mlir::Pass> gpu_runtime::createGPUExPass() {
  return std::make_unique<GPUExPass>();
}

std::unique_ptr<mlir::Pass> gpu_runtime::createGenDeviceFuncsPass() {
  return std::make_unique<GenDeviceFuncsPass>();
}

std::unique_ptr<mlir::Pass> gpu_runtime::createParallelLoopGPUMappingPass() {
  return std::make_unique<ParallelLoopGPUMappingPass>();
}

std::unique_ptr<mlir::Pass> gpu_runtime::createTileParallelLoopsForGPUPass() {
  return std::make_unique<TileParallelLoopsForGPUPass>();
}

std::unique_ptr<mlir::Pass> gpu_runtime::createTruncateF64ForGPUPass() {
  return std::make_unique<TruncateF64ForGPUPass>();
}

std::unique_ptr<mlir::Pass> gpu_runtime::createInsertGPUGlobalReducePass() {
  return std::make_unique<InsertGPUGlobalReducePass>();
}

std::unique_ptr<mlir::Pass> gpu_runtime::createLowerGPUGlobalReducePass() {
  return std::make_unique<LowerGPUGlobalReducePass>();
}

std::unique_ptr<mlir::Pass> gpu_runtime::createSortParallelLoopsForGPU() {
  return std::make_unique<SortParallelLoosForGPU>();
}

std::unique_ptr<Pass> gpu_runtime::createCreateGPUAllocPass() {
  return std::make_unique<CreateGPUAllocPass>();
}
