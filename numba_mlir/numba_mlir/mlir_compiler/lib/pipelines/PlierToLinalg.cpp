// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "pipelines/PlierToLinalg.hpp"

#include <mlir/Analysis/AliasAnalysis.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Arith/Transforms/Passes.h>
#include <mlir/Dialect/Arith/Utils/Utils.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Bufferization/Pipelines/Passes.h>
#include <mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h>
#include <mlir/Dialect/Bufferization/Transforms/Bufferize.h>
#include <mlir/Dialect/Bufferization/Transforms/Passes.h>
#include <mlir/Dialect/Complex/IR/Complex.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/Passes.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Linalg/Passes.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/Math/Transforms/Passes.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/SCF/Transforms/Passes.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Tensor/Transforms/Passes.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/Dominance.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/LoopInvariantCodeMotionUtils.h>
#include <mlir/Transforms/Passes.h>

#include "pipelines/PlierToScf.hpp"
#include "pipelines/PlierToStd.hpp"
#include "pipelines/PreLowSimplifications.hpp"

#include "numba/Analysis/AliasAnalysis.hpp"
#include "numba/Compiler/PipelineRegistry.hpp"
#include "numba/Conversion/NtensorToLinalg.hpp"
#include "numba/Conversion/NtensorToMemref.hpp"
#include "numba/Dialect/gpu_runtime/IR/GpuRuntimeOps.hpp"
#include "numba/Dialect/ntensor/IR/NTensorOps.hpp"
#include "numba/Dialect/ntensor/Transforms/PropagateEnvironment.hpp"
#include "numba/Dialect/ntensor/Transforms/ResolveArrayOps.hpp"
#include "numba/Dialect/numba_util/Dialect.hpp"
#include "numba/Dialect/plier/Dialect.hpp"
#include "numba/Transforms/CanonicalizeReductions.hpp"
#include "numba/Transforms/CastUtils.hpp"
#include "numba/Transforms/CommonOpts.hpp"
#include "numba/Transforms/CompositePass.hpp"
#include "numba/Transforms/CopyRemoval.hpp"
#include "numba/Transforms/ExpandTuple.hpp"
#include "numba/Transforms/FuncTransforms.hpp"
#include "numba/Transforms/InlineUtils.hpp"
#include "numba/Transforms/LoopUtils.hpp"
#include "numba/Transforms/MakeSignless.hpp"
#include "numba/Transforms/MemoryRewrites.hpp"
#include "numba/Transforms/PipelineUtils.hpp"
#include "numba/Transforms/PromoteBoolMemref.hpp"
#include "numba/Transforms/PromoteToParallel.hpp"
#include "numba/Transforms/RewriteWrapper.hpp"
#include "numba/Transforms/ShapeIntegerRangePropagation.hpp"
#include "numba/Transforms/TypeConversion.hpp"
#include "numba/Transforms/UpliftMath.hpp"

#include "BasePipeline.hpp"
#include "NumpyResolver.hpp"
#include "PyLinalgResolver.hpp"

#include <cctype>

namespace {
static numba::util::EnvironmentRegionOp
isInsideParallelRegion(mlir::Operation *op) {
  assert(op && "Invalid op");
  while (true) {
    auto region = op->getParentOfType<numba::util::EnvironmentRegionOp>();
    if (!region)
      return nullptr;

    if (mlir::isa<numba::util::ParallelAttr>(region.getEnvironment()))
      return region;

    op = region;
  }
}

static numba::util::EnvironmentRegionOp
isInsideAtomicRegion(mlir::Operation *op) {
  assert(op && "Invalid op");
  while (true) {
    auto region = op->getParentOfType<numba::util::EnvironmentRegionOp>();
    if (!region)
      return nullptr;

    if (mlir::isa<numba::util::AtomicAttr>(region.getEnvironment()))
      return region;

    op = region;
  }
}

static int64_t getOptLevel(mlir::Operation *op) {
  assert(op);
  auto attr = op->getAttr(numba::util::attributes::getOptLevelName())
                  .dyn_cast_or_null<mlir::IntegerAttr>();
  if (!attr)
    return 0;

  return std::max(static_cast<int64_t>(0), attr.getInt());
}

static bool optimizeSimpleLoads(mlir::Operation *op) {
  bool changed = false;

  mlir::DominanceInfo dom;
  op->walk([&](mlir::memref::LoadOp load) {
    auto memref = load.getMemRef();
    auto src = memref.getDefiningOp();
    if (!mlir::isa_and_nonnull<mlir::memref::AllocOp, mlir::memref::AllocaOp>(
            src))
      return;

    mlir::memref::StoreOp store;
    for (auto user : src->getUsers()) {
      if (mlir::isa<mlir::memref::DeallocOp, mlir::memref::LoadOp>(user))
        continue;

      auto reshape = mlir::dyn_cast<mlir::memref::ReshapeOp>(user);
      if (reshape && reshape.getShape() == memref)
        continue;

      if (mlir::isa<mlir::memref::DeallocOp, mlir::memref::LoadOp>(user))
        continue;

      auto newStore = mlir::dyn_cast<mlir::memref::StoreOp>(user);
      if (!newStore || !dom.properlyDominates(newStore, load))
        return;

      if (load.getType() != newStore.getValueToStore().getType() ||
          load.getIndices() != newStore.getIndices())
        continue;

      if (store && dom.properlyDominates(store, newStore))
        continue;

      store = newStore;
    }

    if (!store)
      return;

    load.getResult().replaceAllUsesWith(store.getValueToStore());
    load->erase();
    changed = true;
  });

  return changed;
}

static mlir::LogicalResult applyOptimizations(
    mlir::func::FuncOp op, const mlir::FrozenRewritePatternSet &patterns,
    mlir::AnalysisManager am,
    llvm::function_ref<mlir::LogicalResult(mlir::func::FuncOp)> additionalOpts =
        nullptr) {
  bool repeat = false;
  do {
    repeat = false;
    (void)mlir::applyPatternsAndFoldGreedily(op, patterns);

    auto memOptRes = numba::optimizeMemoryOps(am);
    if (!memOptRes)
      return op.emitError() << "Failed to build memssa analysis";

    if (mlir::succeeded(*memOptRes))
      repeat = true;

    if (optimizeSimpleLoads(op))
      repeat = true;

    if (additionalOpts && mlir::succeeded(additionalOpts(op)))
      repeat = true;

    if (repeat)
      am.invalidate({});

  } while (repeat);
  return mlir::success();
}

static void rerunScfPipeline(mlir::Operation *op) {
  assert(nullptr != op);
  auto marker =
      mlir::StringAttr::get(op->getContext(), plierToScfPipelineName());
  auto mod = op->getParentOfType<mlir::ModuleOp>();
  assert(nullptr != mod);
  numba::addPipelineJumpMarker(mod, marker);
}

static std::optional<mlir::Type> isUniTuple(mlir::TupleType type) {
  auto count = type.size();
  if (count == 0)
    return std::nullopt;
  auto elemType = type.getType(0);
  for (auto i : llvm::seq<size_t>(1, count)) {
    if (type.getType(i) != elemType)
      return std::nullopt;
  }
  return elemType;
}

static std::optional<mlir::Type> isUniTuple(mlir::Type type) {
  auto tupleType = type.dyn_cast<mlir::TupleType>();
  if (!tupleType)
    return std::nullopt;

  return isUniTuple(tupleType);
}

static void genCopy(mlir::OpBuilder &builder, mlir::Location loc,
                    mlir::Value src, mlir::Value dst) {
  auto srcType = src.getType().cast<mlir::ShapedType>();
  auto dstType = dst.getType().cast<mlir::ShapedType>();
  assert(srcType.getRank() == dstType.getRank());
  assert(srcType.getElementType() == dstType.getElementType());
  auto rank = srcType.getRank();

  auto affineMap =
      mlir::AffineMap::getMultiDimIdentityMap(rank, builder.getContext());
  const mlir::AffineMap maps[] = {
      affineMap,
      affineMap,
  };

  llvm::SmallVector<mlir::utils::IteratorType> iterators(
      rank, mlir::utils::IteratorType::parallel);

  auto bodyBuilder = [](mlir::OpBuilder &b, mlir::Location l,
                        mlir::ValueRange args) {
    assert(args.size() == 2);
    b.create<mlir::linalg::YieldOp>(l, args.front());
  };
  builder.create<mlir::linalg::GenericOp>(loc, src, dst, maps, iterators,
                                          bodyBuilder);
}

struct CleanupLoads : public mlir::OpRewritePattern<mlir::memref::LoadOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::LoadOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto block = op->getBlock();
    auto it = mlir::Block::iterator(op);
    if (it == block->begin())
      return mlir::failure();

    --it;
    auto store = mlir::dyn_cast<mlir::memref::StoreOp>(*it);
    if (!store)
      return mlir::failure();

    if (store.getMemref() != op.getMemref() ||
        store.getIndices() != op.getIndices())
      return mlir::failure();

    rewriter.replaceOp(op, store.getValue());
    return mlir::success();
  }
};

static auto getSizes(mlir::OpBuilder &builder, mlir::Location loc,
                     mlir::Value src) {
  auto shape = src.getType().cast<mlir::ShapedType>().getShape();

  assert(mlir::isa<mlir::MemRefType>(src.getType()));
  llvm::SmallVector<mlir::OpFoldResult> sizes(shape.size());
  for (auto &&[i, dim] : llvm::enumerate(shape)) {
    if (mlir::ShapedType::isDynamic(dim)) {
      sizes[i] = builder.createOrFold<mlir::memref::DimOp>(loc, src, i);
    } else {
      sizes[i] = builder.getIndexAttr(dim);
    }
  }
  return sizes;
}

static auto
computeIdentityStrides(mlir::OpBuilder &builder, mlir::Location loc,
                       llvm::ArrayRef<int64_t> shape,
                       llvm::ArrayRef<mlir::OpFoldResult> dynamicSizes) {
  auto rank = shape.size();
  assert(dynamicSizes.size() == rank);

  int64_t stride = 1;
  llvm::SmallVector<mlir::OpFoldResult> expectedStrides(rank);
  mlir::Value runningStride =
      builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
  for (auto ii = rank; ii-- > 0;) {
    auto i = static_cast<unsigned>(ii);
    expectedStrides[i] = runningStride;

    int64_t size = shape[i];
    if (size == 0)
      continue;

    bool useSizeAsStride = (stride == 1);
    if (size == mlir::ShapedType::kDynamic)
      stride = mlir::ShapedType::kDynamic;
    if (stride != mlir::ShapedType::kDynamic)
      stride *= size;

    auto sizeVal =
        mlir::getValueOrCreateConstantIndexOp(builder, loc, dynamicSizes[i]);
    if (useSizeAsStride)
      runningStride = sizeVal;
    else if (stride == mlir::ShapedType::kDynamic)
      runningStride =
          builder.create<mlir::arith::MulIOp>(loc, runningStride, sizeVal);
    else
      runningStride = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
  }

  return expectedStrides;
}

struct ReshapeChangeLayout
    : public mlir::OpRewritePattern<numba::util::ReshapeOp> {

  // Set high benefit, so it will run earlier than ReshapeToReinterpret.
  ReshapeChangeLayout(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<numba::util::ReshapeOp>(context,
                                                       /*benefit*/ 10) {}

  mlir::LogicalResult
  matchAndRewrite(numba::util::ReshapeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto cl = op.getSource().getDefiningOp<numba::util::ChangeLayoutOp>();
    if (!cl)
      return mlir::failure();

    auto src = cl.getSource();
    auto srcType = mlir::dyn_cast<mlir::MemRefType>(src.getType());
    if (!srcType)
      return mlir::failure();

    auto dstType = mlir::dyn_cast<mlir::MemRefType>(op.getSource().getType());
    if (!dstType)
      return mlir::failure();

    if (srcType.getRank() != dstType.getRank())
      return mlir::failure();

    auto rank = static_cast<unsigned>(dstType.getRank());
    if (rank == 0)
      return mlir::failure();

    int64_t offset;
    llvm::SmallVector<int64_t> strides;
    if (mlir::failed(mlir::getStridesAndOffset(dstType, strides, offset)))
      return mlir::failure();

    auto loc = op.getLoc();
    auto sizesVals = getSizes(rewriter, loc, src);
    auto expectedStrides =
        computeIdentityStrides(rewriter, loc, srcType.getShape(), sizesVals);

    mlir::OpFoldResult offsetVal = rewriter.getIndexAttr(offset);

    llvm::SmallVector<mlir::OpFoldResult> stridesVals(rank);

    auto metadata =
        rewriter.create<mlir::memref::ExtractStridedMetadataOp>(loc, src);
    auto actualStrides = metadata.getStrides();

    mlir::Value cmp;
    for (auto i : llvm::seq(0u, rank)) {
      if (mlir::ShapedType::isDynamic(strides[i])) {
        stridesVals[i] = expectedStrides[i].get<mlir::Value>();
      } else {
        stridesVals[i] = rewriter.getIndexAttr(strides[i]);
      }

      auto cmpTemp = rewriter.createOrFold<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::eq,
          expectedStrides[i].get<mlir::Value>(), actualStrides[i]);

      if (i == 0) {
        cmp = cmpTemp;
      } else {
        cmp = rewriter.createOrFold<mlir::arith::AndIOp>(loc, cmp, cmpTemp);
      }
    }

    auto trueBody = [&](mlir::OpBuilder &builder, mlir::Location loc) {
      mlir::Value flat = builder.create<numba::util::MemrefApplyOffsetOp>(
          loc, src.getType(), src);

      mlir::Value res = builder.create<mlir::memref::ReinterpretCastOp>(
          loc, dstType, flat, offsetVal, sizesVals, stridesVals);
      builder.create<mlir::scf::YieldOp>(loc, res);
    };
    auto falseBody = [&](mlir::OpBuilder &builder, mlir::Location loc) {
      llvm::SmallVector<mlir::Value> sizes;
      sizes.reserve(rank);
      auto shape = dstType.getShape();
      for (auto i : llvm::seq(0u, rank))
        if (mlir::ShapedType::isDynamic(shape[i]))
          sizes.emplace_back(sizesVals[i].get<mlir::Value>());

      auto res = builder.create<mlir::memref::AllocOp>(loc, dstType, sizes)
                     .getResult();
      genCopy(rewriter, loc, src, res);
      builder.create<mlir::scf::YieldOp>(loc, res);
    };

    auto res = rewriter.create<mlir::scf::IfOp>(loc, cmp, trueBody, falseBody)
                   .getResult(0);
    rewriter.replaceOpWithNewOp<numba::util::ReshapeOp>(op, op.getType(), res,
                                                        op.getShape());
    return mlir::success();
  }
};

struct ReshapeToReinterpret
    : public mlir::OpRewritePattern<numba::util::ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(numba::util::ReshapeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto src = op.getSource();
    auto srcType = mlir::dyn_cast<mlir::MemRefType>(src.getType());
    if (!srcType || !srcType.getLayout().isIdentity())
      return mlir::failure();

    auto dstType = mlir::dyn_cast<mlir::MemRefType>(op.getResult().getType());
    if (!dstType || !dstType.getLayout().isIdentity())
      return mlir::failure();

    llvm::SmallVector<mlir::OpFoldResult> shape = op.getShape();
    auto loc = op.getLoc();
    auto strides =
        computeIdentityStrides(rewriter, loc, dstType.getShape(), shape);

    mlir::OpFoldResult offset = rewriter.getIndexAttr(0);
    rewriter.replaceOpWithNewOp<mlir::memref::ReinterpretCastOp>(
        op, dstType, src, offset, shape, strides);
    return mlir::success();
  }
};

struct MakeStridedLayoutPass
    : public mlir::PassWrapper<MakeStridedLayoutPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MakeStridedLayoutPass)

  void runOnOperation() override {
    auto context = &getContext();
    auto mod = getOperation();

    mlir::OpBuilder builder(mod);
    llvm::SmallVector<mlir::Type> newResTypes;
    for (auto func : mod.getOps<mlir::func::FuncOp>()) {
      auto funcType = func.getFunctionType();
      auto argTypes = funcType.getInputs();
      auto resTypes = funcType.getResults();

      newResTypes.assign(resTypes.begin(), resTypes.end());
      auto &body = func.getBody();
      bool hasBody = !body.empty();
      if (hasBody)
        builder.setInsertionPointToStart(&body.front());

      for (auto &&[i, type] : llvm::enumerate(resTypes)) {
        auto memrefType = type.dyn_cast<mlir::MemRefType>();
        if (!memrefType || !memrefType.getLayout().isIdentity())
          continue;

        auto rank = static_cast<unsigned>(memrefType.getRank());
        auto makeShape = [&](int64_t val) {
          return llvm::SmallVector<int64_t>(rank, val);
        };
        auto strideVal = mlir::ShapedType::kDynamic;
        auto layout = mlir::StridedLayoutAttr::get(context, strideVal,
                                                   makeShape(strideVal));
        auto newmemrefType =
            mlir::MemRefType::get(makeShape(mlir::ShapedType::kDynamic),
                                  memrefType.getElementType(), layout);
        newResTypes[i] = newmemrefType;
      }

      auto newFuncType =
          mlir::FunctionType::get(&getContext(), argTypes, newResTypes);
      if (newFuncType != funcType) {
        func.setType(newFuncType);
        func.walk([&](mlir::func::ReturnOp ret) {
          builder.setInsertionPoint(ret);
          auto count = static_cast<unsigned>(newResTypes.size());
          for (auto i : llvm::seq(0u, count)) {
            auto arg = ret.getOperand(i);
            auto newType = newResTypes[i];
            if (arg.getType() != newType) {
              assert(arg.getType().isa<mlir::MemRefType>());
              assert(newType.isa<mlir::MemRefType>());
              auto newArg = builder.createOrFold<mlir::memref::CastOp>(
                  ret.getLoc(), newType, arg);
              ret.setOperand(i, newArg);
            }
          }
        });
        auto funcUses = mlir::SymbolTable::getSymbolUses(func, mod);
        if (funcUses) {
          for (auto use : *funcUses) {
            auto call = mlir::dyn_cast<mlir::func::CallOp>(use.getUser());
            if (!call) {
              use.getUser()->emitError("Unsupported func user");
              return signalPassFailure();
            }
            auto loc = call.getLoc();

            builder.setInsertionPointAfter(call);
            assert(newResTypes.size() == call.getNumResults());
            auto numResults = call.getNumResults();
            for (auto i : llvm::seq(0u, numResults)) {
              auto res = call.getResult(i);
              auto oldType = res.getType();
              auto newType = newResTypes[i];
              if (oldType != newType) {
                assert(oldType.isa<mlir::MemRefType>());
                assert(newType.isa<mlir::MemRefType>());
                res.setType(newType);
                auto newRes = builder.create<numba::util::ChangeLayoutOp>(
                    loc, oldType, res);
                res.replaceAllUsesExcept(newRes, newRes);
              }
            }
          }
        }
      }
    }
  }
};

struct ChangeLayoutReturn
    : public mlir::OpRewritePattern<mlir::func::ReturnOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::func::ReturnOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op.getOperands().empty())
      return mlir::failure();

    auto func = op->getParentOfType<mlir::func::FuncOp>();
    if (!func || !func.isPrivate() || !llvm::hasSingleElement(func.getBody()))
      return mlir::failure();

    auto mod = func->getParentOfType<mlir::ModuleOp>();
    if (!mod)
      return mlir::failure();

    auto funcUses = mlir::SymbolTable::getSymbolUses(func, mod);
    if (!funcUses)
      return mlir::failure();

    for (auto use : *funcUses)
      if (!mlir::isa<mlir::func::CallOp>(use.getUser()))
        return mlir::failure();

    auto loc = op.getLoc();
    auto args = op.getOperands();
    auto count = static_cast<unsigned>(args.size());
    llvm::SmallVector<mlir::Value> newArgs(args.begin(), args.end());
    llvm::SmallVector<int64_t> shape;

    bool changed = false;
    for (auto i : llvm::seq(0u, count)) {
      auto arg = args[i];
      auto retType = mlir::dyn_cast<mlir::MemRefType>(arg.getType());
      if (!retType)
        continue;

      auto src = [&]() -> mlir::Value {
        if (auto cast = arg.getDefiningOp<mlir::memref::CastOp>())
          return cast.getSource();

        if (auto cast = arg.getDefiningOp<numba::util::ChangeLayoutOp>())
          return cast.getSource();

        return nullptr;
      }();

      if (!src)
        continue;

      auto srcType = mlir::cast<mlir::MemRefType>(src.getType());
      assert(srcType.getElementType() == retType.getElementType());

      auto srcLayout = srcType.getLayout();
      auto srcShape = srcType.getShape();
      auto dstShape = retType.getShape();
      assert(srcShape.size() == dstShape.size());
      auto rank = static_cast<unsigned>(srcShape.size());
      shape.resize(rank);
      for (auto j : llvm::seq(0u, rank)) {
        if (!mlir::ShapedType::isDynamic(dstShape[j])) {
          shape[j] = dstShape[j];
        } else if (!mlir::ShapedType::isDynamic(srcShape[j])) {
          shape[j] = srcShape[j];
        } else {
          shape[j] = mlir::ShapedType::kDynamic;
        }
      }

      auto newType = mlir::MemRefType::get(shape, srcType.getElementType(),
                                           srcLayout, srcType.getMemorySpace());
      if (newType == retType)
        continue;

      auto newArg = rewriter.create<mlir::memref::CastOp>(loc, newType, src);
      newArgs[i] = newArg;
      changed = true;
    }

    if (!changed)
      return mlir::failure();

    rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op, newArgs);

    auto newFuncType = [&]() {
      auto origType = func.getFunctionType();
      mlir::ValueRange r(newArgs);
      return mlir::FunctionType::get(getContext(), origType.getInputs(),
                                     r.getTypes());
    }();

    rewriter.modifyOpInPlace(func, [&]() {
      func.setFunctionTypeAttr(mlir::TypeAttr::get(newFuncType));
    });

    llvm::SmallVector<mlir::func::CallOp> calls;
    for (auto use : *funcUses) {
      auto call = mlir::cast<mlir::func::CallOp>(use.getUser());
      calls.emplace_back(call);
    }

    for (auto call : calls) {
      rewriter.setInsertionPoint(call);
      auto callLoc = call->getLoc();
      auto oldResults = call.getResults();
      auto newResults =
          rewriter.create<mlir::func::CallOp>(callLoc, func, call.getOperands())
              .getResults();
      newArgs.assign(newResults.begin(), newResults.end());
      for (auto i : llvm::seq(0u, count)) {
        auto oldType = oldResults[i].getType();
        auto newType = newArgs[i].getType();
        if (oldType != newType)
          newArgs[i] = rewriter.create<numba::util::ChangeLayoutOp>(
              callLoc, oldType, newArgs[i]);
      }
      rewriter.replaceOp(call, newArgs);
    }

    return mlir::success();
  }
};

static int64_t getMostDynamicDim(int64_t dim1, int64_t dim2) {
  if (dim1 == dim2)
    return dim1;

  return mlir::ShapedType::kDynamic;
}

static std::optional<mlir::MemRefLayoutAttrInterface>
getMostDynamicLayout(mlir::MemRefLayoutAttrInterface l1,
                     mlir::MemRefLayoutAttrInterface l2) {
  if (l1.isIdentity())
    return l2;

  if (l2.isIdentity())
    return l1;

  auto strided1 = mlir::dyn_cast<mlir::StridedLayoutAttr>(l1);
  if (!strided1)
    return std::nullopt;

  auto strided2 = mlir::dyn_cast<mlir::StridedLayoutAttr>(l2);
  if (!strided2)
    return std::nullopt;

  auto strides1 = strided1.getStrides();
  auto strides2 = strided2.getStrides();
  if (strides1.size() != strides2.size())
    return std::nullopt;

  auto newOffset =
      getMostDynamicDim(strided1.getOffset(), strided2.getOffset());

  llvm::SmallVector<int64_t> newStrides;
  for (auto &&[i, it] : llvm::enumerate(llvm::zip(strides1, strides2))) {
    auto &&[s1, s2] = it;
    newStrides[i] = getMostDynamicDim(s1, s2);
  }
  return mlir::StridedLayoutAttr::get(l1.getContext(), newOffset, newStrides);
}

struct ChangeLayoutCall : public mlir::OpRewritePattern<mlir::func::CallOp> {
  // Set benfit lower than canonicalization patterns.
  ChangeLayoutCall(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<mlir::func::CallOp>(context, /*benefit*/ 0) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::func::CallOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    if (!mod)
      return mlir::failure();

    mlir::func::FuncOp func =
        mod.lookupSymbol<mlir::func::FuncOp>(op.getCalleeAttr());
    if (!func || func.isDeclaration() || !func.isPrivate())
      return mlir::failure();

    bool checked = false;
    llvm::SmallVector<mlir::func::CallOp> otherCalls;
    auto getUsers = [&]() -> bool {
      if (checked)
        return true;

      auto uses = mlir::SymbolTable::getSymbolUses(func, mod);
      if (!uses)
        return false;

      for (auto &use : *uses) {
        auto call = mlir::dyn_cast<mlir::func::CallOp>(use.getUser());
        if (!call)
          return false;

        if (call == op)
          continue;

        otherCalls.emplace_back(call);
      }
      checked = true;
      return true;
    };

    mlir::ValueRange args = op.getOperands();

    auto loc = op.getLoc();
    llvm::SmallVector<mlir::Value> newArgs(args.begin(), args.end());
    for (auto &&[i, arg] : llvm::enumerate(args)) {
      auto dstType = mlir::dyn_cast<mlir::MemRefType>(arg.getType());
      if (!dstType)
        continue;

      auto changeLayout = arg.getDefiningOp<numba::util::ChangeLayoutOp>();
      if (!changeLayout)
        continue;

      mlir::Value src = changeLayout.getSource();
      auto srcType = mlir::cast<mlir::MemRefType>(src.getType());
      auto newLayout =
          getMostDynamicLayout(srcType.getLayout(), dstType.getLayout());
      if (!newLayout || *newLayout == dstType.getLayout())
        continue;

      if (!getUsers())
        return mlir::failure();

      auto newType =
          mlir::MemRefType::get(dstType.getShape(), dstType.getElementType(),
                                *newLayout, dstType.getMemorySpace());
      if (srcType != newType)
        src = rewriter.create<mlir::memref::CastOp>(loc, newType, src);

      newArgs[i] = src;
    }

    if (!checked)
      return mlir::failure();

    rewriter.replaceOpWithNewOp<mlir::func::CallOp>(op, func, newArgs);

    auto newArgTypes =
        llvm::map_to_vector(newArgs, [](auto arg) { return arg.getType(); });
    auto funcType = func.getFunctionType();
    auto newFuncType = funcType.clone(newArgTypes, funcType.getResults());

    mlir::OpBuilder::InsertionGuard g(rewriter);
    rewriter.modifyOpInPlace(func, [&]() {
      func.setType(newFuncType);
      assert(!func.getFunctionBody().empty());
      auto &block = func.getFunctionBody().front();
      rewriter.setInsertionPointToStart(&block);
      assert(block.getNumArguments() == newArgTypes.size());
      for (auto &&[arg, newType] :
           llvm::zip(block.getArguments(), newArgTypes)) {
        if (arg.getType() == newType)
          continue;

        auto oldType = arg.getType();
        arg.setType(newType);
        auto cast =
            rewriter.create<numba::util::ChangeLayoutOp>(loc, oldType, arg);
        rewriter.replaceAllUsesExcept(arg, cast.getResult(), cast);
      }
    });

    for (auto call : otherCalls) {
      auto loc = call.getLoc();
      rewriter.setInsertionPoint(call);
      auto args = call.getOperands();
      for (auto &&[i, it] : llvm::enumerate(llvm::zip(args, newArgTypes))) {
        auto &&[arg, newType] = it;
        if (arg.getType() == newType) {
          newArgs[i] = arg;
          continue;
        }

        newArgs[i] = rewriter.create<mlir::memref::CastOp>(loc, newType, arg);
      }

      rewriter.replaceOpWithNewOp<mlir::func::CallOp>(call, func, newArgs);
    }
    return mlir::success();
  }
};

struct OptimizeStridedLayoutPass
    : public mlir::PassWrapper<OptimizeStridedLayoutPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OptimizeStridedLayoutPass)

  void runOnOperation() override {
    auto *context = &getContext();
    mlir::RewritePatternSet patterns(context);

    numba::populateCanonicalizationPatterns(patterns);

    patterns.insert<ChangeLayoutReturn, ChangeLayoutCall>(context);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                        std::move(patterns))))
      return signalPassFailure();
  }
};

struct FinalizeStridedLayoutPass
    : public mlir::PassWrapper<FinalizeStridedLayoutPass,
                               mlir::OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FinalizeStridedLayoutPass)

  void runOnOperation() override;
};

void FinalizeStridedLayoutPass::runOnOperation() {
  auto *context = &getContext();
  auto op = getOperation();
  mlir::RewritePatternSet patterns(context);

  patterns.insert<ReshapeChangeLayout, ReshapeToReinterpret, CleanupLoads>(
      context);

  if (mlir::failed(mlir::applyPatternsAndFoldGreedily(op, std::move(patterns))))
    return signalPassFailure();

  op->walk([&](numba::util::ChangeLayoutOp cl) {
    cl.emitError("Layout change failed");
    signalPassFailure();
  });
}

static mlir::Value convertScalarType(mlir::OpBuilder &builder,
                                     mlir::Location loc, mlir::Value val,
                                     mlir::Type dstType) {
  auto srcType = val.getType();
  if (numba::canConvert(srcType, dstType))
    val = numba::doConvert(builder, loc, val, dstType);

  return val;
}

struct GetitemToNtensor : public mlir::OpConversionPattern<plier::GetItemOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::GetItemOp op, plier::GetItemOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto src = adaptor.getValue();
    auto srcType = src.getType().dyn_cast<numba::ntensor::NTensorType>();
    if (!srcType)
      return mlir::failure();

    auto converter = getTypeConverter();
    assert(converter);
    auto resultType = converter->convertType(op.getType());
    if (!resultType)
      return mlir::failure();

    auto index = adaptor.getIndex();

    rewriter.replaceOpWithNewOp<numba::ntensor::GetitemOp>(op, resultType, src,
                                                           index);
    return mlir::success();
  }
};

struct SetitemToNtensor : public mlir::OpConversionPattern<plier::SetItemOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::SetItemOp op, plier::SetItemOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto src = adaptor.getTarget();
    auto srcType = src.getType().dyn_cast<numba::ntensor::NTensorType>();
    if (!srcType)
      return mlir::failure();

    auto index = adaptor.getIndex();
    auto value = convertScalarType(rewriter, op->getLoc(), adaptor.getValue(),
                                   srcType.getElementType());

    rewriter.replaceOpWithNewOp<numba::ntensor::SetitemOp>(op, src, index,
                                                           value);
    return mlir::success();
  }
};

struct NtensorGetitemToNtensor
    : public mlir::OpConversionPattern<numba::ntensor::GetitemOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(numba::ntensor::GetitemOp op,
                  numba::ntensor::GetitemOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    assert(converter);
    auto resultType = converter->convertType(op.getType());
    if (!resultType)
      return mlir::failure();

    rewriter.replaceOpWithNewOp<numba::ntensor::GetitemOp>(
        op, resultType, adaptor.getSource(), adaptor.getIndex());
    return mlir::success();
  }
};

struct NtensorSetitemToNtensor
    : public mlir::OpConversionPattern<numba::ntensor::SetitemOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(numba::ntensor::SetitemOp op,
                  numba::ntensor::SetitemOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<numba::ntensor::SetitemOp>(
        op, adaptor.getSource(), adaptor.getIndex(), adaptor.getValue());
    return mlir::success();
  }
};

struct UnaryToNtensor : public mlir::OpConversionPattern<plier::UnaryOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::UnaryOp op, plier::UnaryOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto val = adaptor.getValue();
    if (!val.getType().isa<numba::ntensor::NTensorType>())
      return mlir::failure();

    auto converter = getTypeConverter();
    assert(converter);
    auto resultType = converter->convertType(op.getType());
    if (!resultType)
      return mlir::failure();

    auto opName = op.getOp();
    rewriter.replaceOpWithNewOp<numba::ntensor::UnaryOp>(op, resultType, val,
                                                         opName);
    return mlir::success();
  }
};

struct BinopToNtensor : public mlir::OpConversionPattern<plier::BinOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::BinOp op, plier::BinOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    if (!lhs.getType().isa<numba::ntensor::NTensorType>() &&
        !rhs.getType().isa<numba::ntensor::NTensorType>())
      return mlir::failure();

    auto converter = getTypeConverter();
    assert(converter);
    auto resultType = converter->convertType(op.getType());
    if (!resultType)
      return mlir::failure();

    auto opName = op.getOp();
    rewriter.replaceOpWithNewOp<numba::ntensor::BinaryOp>(op, resultType, lhs,
                                                          rhs, opName);
    return mlir::success();
  }
};

struct InplaceBinopToNtensor
    : public mlir::OpConversionPattern<plier::InplaceBinOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::InplaceBinOp op, plier::InplaceBinOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    if (!lhs.getType().isa<numba::ntensor::NTensorType>() &&
        !rhs.getType().isa<numba::ntensor::NTensorType>())
      return mlir::failure();

    auto converter = getTypeConverter();
    assert(converter);
    auto resultType = converter->convertType(op.getType());
    if (!resultType)
      return mlir::failure();

    auto loc = op.getLoc();
    auto opName = op.getOp();
    mlir::Value res = rewriter.create<numba::ntensor::BinaryOp>(
        loc, resultType, lhs, rhs, opName);
    rewriter.create<numba::ntensor::CopyOp>(loc, res, lhs);
    rewriter.replaceOp(op, lhs);
    return mlir::success();
  }
};

struct BuildSliceToNtensor
    : public mlir::OpConversionPattern<plier::BuildSliceOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::BuildSliceOp op, plier::BuildSliceOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto indexType = rewriter.getIndexType();

    auto loc = op.getLoc();
    auto doCast = [&](mlir::Value val) -> std::optional<mlir::Value> {
      if (numba::canConvert(val.getType(), indexType))
        return numba::doConvert(rewriter, loc, val, indexType);

      return std::nullopt;
    };

    auto isNone = [](mlir::Value val) {
      return val.getType().isa<mlir::NoneType>();
    };

    auto getVal = [&](mlir::Value orig,
                      mlir::Value converted) -> std::optional<mlir::Value> {
      if (isNone(orig))
        return mlir::Value{};

      return doCast(orig);
    };

    auto begin = getVal(op.getBegin(), adaptor.getBegin());
    auto end = getVal(op.getEnd(), adaptor.getEnd());
    auto step = getVal(op.getStep(), adaptor.getStep());
    if (!begin || !end || !step)
      return mlir::failure();

    rewriter.replaceOpWithNewOp<numba::ntensor::BuildSliceOp>(op, *begin, *end,
                                                              *step);
    return mlir::success();
  }
};

static bool isBoundFunc(mlir::Type type) {
  return mlir::isa<plier::BoundFunctionType>(type);
}

struct NumpyCallsToNtensor : public mlir::OpConversionPattern<plier::PyCallOp> {
  NumpyCallsToNtensor(mlir::TypeConverter &converter, mlir::MLIRContext *ctx,
                      NumpyResolver &r)
      : OpConversionPattern(converter, ctx), resolver(r) {}

  mlir::LogicalResult
  matchAndRewrite(plier::PyCallOp op, plier::PyCallOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto func = adaptor.getFunc();
    if (!func || !mlir::isa<plier::FunctionType, plier::BoundFunctionType>(
                     func.getType()))
      return mlir::failure();

    auto getAttr = func.getDefiningOp<plier::GetattrOp>();
    bool isAttr = getAttr && isBoundFunc(func.getType());

    std::string funcName;
    if (isAttr) {
      if (!mlir::isa<numba::ntensor::NTensorType>(getAttr.getValue().getType()))
        return mlir::failure();

      funcName = ("array." + getAttr.getName()).str();
    } else {
      funcName = op.getFuncName().str();
    }

    if (!resolver.hasFunc(funcName))
      return mlir::failure();

    auto converter = getTypeConverter();
    assert(converter);

    llvm::SmallVector<mlir::Type> resTypes;
    if (mlir::failed(converter->convertTypes(op->getResultTypes(), resTypes)))
      return mlir::failure();

    llvm::SmallVector<mlir::Value> args;
    llvm::SmallVector<mlir::Attribute> argNames;
    auto srcArgs = adaptor.getArgs();
    auto srcKwArgs = adaptor.getKwargs();
    auto srcKwNames = adaptor.getKwNames();
    auto totalCount =
        srcArgs.size() + srcKwArgs.size() + static_cast<size_t>(isAttr);
    args.reserve(totalCount);
    argNames.reserve(totalCount);

    auto emptyStrAttr = rewriter.getStringAttr("");
    if (isAttr) {
      auto val = rewriter.getRemappedValue(getAttr.getValue());
      args.emplace_back(val);
      argNames.emplace_back(emptyStrAttr);
    }

    args.append(srcArgs.begin(), srcArgs.end());
    argNames.append(srcArgs.size(), emptyStrAttr);

    args.append(srcKwArgs.begin(), srcKwArgs.end());
    argNames.append(srcKwNames.begin(), srcKwNames.end());

    auto argNamesAttr = rewriter.getArrayAttr(argNames);
    rewriter.replaceOpWithNewOp<numba::ntensor::CallOp>(op, resTypes, args,
                                                        argNamesAttr, funcName);
    return mlir::success();
  }

private:
  NumpyResolver &resolver;
};

struct NumpyAttrsToNtensor
    : public mlir::OpConversionPattern<plier::GetattrOp> {
  NumpyAttrsToNtensor(mlir::TypeConverter &converter, mlir::MLIRContext *ctx,
                      NumpyResolver &r)
      : OpConversionPattern(converter, ctx), resolver(r) {}

  mlir::LogicalResult
  matchAndRewrite(plier::GetattrOp op, plier::GetattrOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto src = adaptor.getValue();
    if (!src.getType().isa<numba::ntensor::NTensorType>())
      return mlir::failure();

    if (isBoundFunc(op.getType()))
      return mlir::failure();

    auto funcName = ("array." + op.getName()).str();
    if (!resolver.hasFunc(funcName))
      return mlir::failure();

    auto converter = getTypeConverter();
    assert(converter);

    auto resultType = converter->convertType(op.getType());
    if (!resultType)
      return mlir::failure();

    auto argNamesAttr = rewriter.getArrayAttr(rewriter.getStringAttr(""));
    rewriter.replaceOpWithNewOp<numba::ntensor::CallOp>(op, resultType, src,
                                                        argNamesAttr, funcName);

    return mlir::success();
  }

private:
  NumpyResolver &resolver;
};

static std::optional<mlir::Value> addElementConversion(mlir::OpBuilder &builder,
                                                       mlir::Location loc,
                                                       mlir::Value srcArray,
                                                       mlir::Type dstType) {
  auto srcType = srcArray.getType().cast<numba::ntensor::NTensorType>();
  auto dstShaped = dstType.cast<mlir::ShapedType>();
  auto srcElementType = srcType.getElementType();
  auto dstElementType = dstShaped.getElementType();
  if (srcElementType == dstElementType)
    return srcArray;

  if (!numba::canConvert(srcElementType, dstElementType))
    return std::nullopt;

  auto dstArrayTupe = numba::ntensor::NTensorType::get(
      dstShaped.getShape(), dstElementType, srcType.getEnvironment(),
      srcType.getLayout());

  rerunScfPipeline(srcArray.getParentRegion()->getParentOp());
  auto bodyBuilder = [&](mlir::OpBuilder &b, mlir::Location l,
                         mlir::ValueRange vals) {
    assert(vals.size() == 1);
    mlir::Value res = numba::doConvert(b, l, vals.front(), dstElementType);
    b.create<numba::ntensor::ElementwiseYieldOp>(l, res);
  };

  return builder
      .create<numba::ntensor::ElementwiseOp>(loc, dstArrayTupe, srcArray,
                                             bodyBuilder)
      .getResult(0);
}

static mlir::Value castType(mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::Value src, mlir::Type dstType) {
  auto srcType = src.getType();
  if (srcType == dstType)
    return src;

  if (srcType.isa<mlir::MemRefType>())
    return builder.create<mlir::memref::CastOp>(loc, dstType, src);

  if (srcType.isa<mlir::RankedTensorType>())
    return builder.create<mlir::tensor::CastOp>(loc, dstType, src);

  if (srcType.isa<numba::ntensor::NTensorType>())
    return builder.create<numba::ntensor::CastOp>(loc, dstType, src);

  llvm_unreachable("Invalid shaped type");
}

static std::optional<mlir::Value> doCast(mlir::OpBuilder &builder,
                                         mlir::Location loc, mlir::Value src,
                                         mlir::Type dstType);

static std::optional<mlir::Value> doTupleCast(mlir::OpBuilder &builder,
                                              mlir::Location loc,
                                              mlir::Value src,
                                              mlir::Type dstType) {
  auto srcTupleType = mlir::dyn_cast<mlir::TupleType>(src.getType());
  if (!srcTupleType)
    return std::nullopt;

  auto dstTupleType = mlir::dyn_cast<mlir::TupleType>(dstType);
  if (!dstTupleType)
    return std::nullopt;

  if (srcTupleType.size() != dstTupleType.size())
    return std::nullopt;

  llvm::SmallVector<mlir::Value> elems;
  elems.reserve(srcTupleType.size());

  for (auto &&[i, it] : llvm::enumerate(
           llvm::zip(srcTupleType.getTypes(), dstTupleType.getTypes()))) {
    auto &&[srcElemType, dstElemType] = it;
    auto srcElem = builder.create<numba::util::TupleExtractOp>(loc, src, i);
    if (srcElemType == dstElemType) {
      elems.emplace_back(srcElem);
      continue;
    }

    auto elem = doCast(builder, loc, srcElem, dstElemType);
    if (!elem)
      return std::nullopt;

    elems.emplace_back(*elem);
  }

  mlir::Value ret = builder.create<numba::util::BuildTupleOp>(loc, elems);
  return ret;
}

static std::optional<mlir::Value> doCast(mlir::OpBuilder &builder,
                                         mlir::Location loc, mlir::Value src,
                                         mlir::Type dstType) {
  auto srcType = src.getType();
  if (srcType == dstType)
    return src;

  if (numba::canConvert(srcType, dstType))
    return numba::doConvert(builder, loc, src, dstType);

  if (auto ret = doTupleCast(builder, loc, src, dstType))
    return ret;

  if (auto srcArrayType = srcType.dyn_cast<numba::ntensor::NTensorType>()) {
    auto dstShapedType = dstType.dyn_cast<mlir::ShapedType>();
    if (!dstShapedType)
      return std::nullopt;

    auto elemConverted = addElementConversion(builder, loc, src, dstShapedType);
    if (!elemConverted)
      return std::nullopt;

    mlir::Value res = *elemConverted;
    if (dstShapedType.isa<mlir::MemRefType>()) {
      auto dstMemrefType = mlir::MemRefType::get(
          srcArrayType.getShape(), dstShapedType.getElementType());
      res = builder.create<numba::ntensor::ToMemrefOp>(loc, dstMemrefType, res);
    } else if (dstShapedType.isa<mlir::RankedTensorType>()) {
      auto dstTensorType = mlir::RankedTensorType::get(
          srcArrayType.getShape(), dstShapedType.getElementType());
      res = builder.create<numba::ntensor::ToTensorCopyOp>(loc, dstTensorType,
                                                           res);
    }

    return castType(builder, loc, res, dstShapedType);
  } else {
    auto srcShapedType = srcType.dyn_cast<mlir::ShapedType>();
    if (!srcShapedType)
      return std::nullopt;

    auto dstArrayType = dstType.dyn_cast<numba::ntensor::NTensorType>();
    if (!dstArrayType)
      return std::nullopt;

    srcArrayType = numba::ntensor::NTensorType::get(
        dstArrayType.getShape(), srcShapedType.getElementType(),
        dstArrayType.getEnvironment(), dstArrayType.getLayout());

    mlir::Value res;
    if (srcShapedType.isa<mlir::MemRefType>()) {
      auto dstMemrefType = mlir::MemRefType::get(
          dstArrayType.getShape(), srcShapedType.getElementType());
      src = castType(builder, loc, src, dstMemrefType);
      res =
          builder.create<numba::ntensor::FromMemrefOp>(loc, srcArrayType, src);
    } else if (srcShapedType.isa<mlir::RankedTensorType>()) {
      auto dstTensorType = mlir::RankedTensorType::get(
          dstArrayType.getShape(), srcShapedType.getElementType());
      src = castType(builder, loc, src, dstTensorType);
      res =
          builder.create<numba::ntensor::FromTensorOp>(loc, srcArrayType, src);
    }
    assert(res && "Expected tensor or memref type.");
    return addElementConversion(builder, loc, res, dstArrayType);
  }
}

static mlir::Value doSafeCast(mlir::OpBuilder &builder, mlir::Location loc,
                              mlir::Value src, mlir::Type dstType) {
  auto res = doCast(builder, loc, src, dstType);
  if (res)
    return *res;

  return builder.create<plier::CastOp>(loc, dstType, src);
}

struct CastsToNtensor : public mlir::OpConversionPattern<plier::CastOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::CastOp op, plier::CastOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value src = adaptor.getValue();
    mlir::Type srcType = src.getType();

    auto converter = getTypeConverter();
    assert(converter);

    auto dstType = converter->convertType(op.getType());
    if (!dstType)
      return mlir::failure();

    if (srcType == dstType) {
      rewriter.replaceOp(op, src);
      return mlir::success();
    }

    auto loc = op.getLoc();
    if (auto res = doCast(rewriter, loc, src, dstType)) {
      rewriter.replaceOp(op, *res);
      return mlir::success();
    }

    if (auto ntensorType = dstType.dyn_cast<numba::ntensor::NTensorType>()) {
      if (!ntensorType.hasStaticShape())
        return mlir::failure();

      auto dstElemType = ntensorType.getElementType();

      if (!numba::canConvert(srcType, dstElemType))
        return mlir::failure();

      src = numba::doConvert(rewriter, loc, src, dstElemType);

      rewriter.replaceOpWithNewOp<numba::ntensor::CreateArrayOp>(
          op, ntensorType, /*dynamicSizes*/ std::nullopt, src);
      return mlir::success();
    }

    return mlir::failure();
  }
};

struct UnitupleExtractToNtensor
    : public mlir::OpConversionPattern<numba::util::TupleExtractOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(numba::util::TupleExtractOp op,
                  numba::util::TupleExtractOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto src = adaptor.getSource();
    auto elemType = isUniTuple(src.getType());
    if (!elemType ||
        !numba::ntensor::NTensorType::isValidElementType(*elemType))
      return mlir::failure();

    auto converter = getTypeConverter();
    assert(converter && "Invalid type converter");

    auto dstType = converter->convertType(op.getType());
    if (!dstType)
      return mlir::failure();

    auto index = adaptor.getIndex();
    rewriter.replaceOpWithNewOp<numba::ntensor::GetitemOp>(op, dstType, src,
                                                           index);
    return mlir::success();
  }
};

struct GetitertConversionPattern
    : public mlir::OpConversionPattern<plier::GetiterOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::GetiterOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    auto converter = getTypeConverter();
    assert(converter && "Invalid type converter");
    if (!mlir::isa<numba::ntensor::NTensorType>(op.getValue().getType()) ||
        !mlir::isa<numba::ntensor::IteratorType>(op.getType()))
      return mlir::failure();

    auto resType = converter->convertType<mlir::TupleType>(op.getType());
    if (!resType || resType.size() != 2)
      return mlir::failure();

    auto loc = op.getLoc();
    mlir::Value begin = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);

    auto iterType = mlir::MemRefType::get({}, rewriter.getIndexType());
    mlir::Value iter = rewriter.create<mlir::memref::AllocOp>(loc, iterType);
    rewriter.create<mlir::memref::StoreOp>(loc, begin, iter);

    mlir::Value rets[] = {iter, adaptor.getValue()};
    mlir::Value ret =
        rewriter.create<numba::util::BuildTupleOp>(loc, resType, rets);
    rewriter.replaceOp(op, ret);
    return mlir::success();
  }
};

struct IternextConversionPattern
    : public mlir::OpConversionPattern<plier::IternextOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::IternextOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    auto converter = getTypeConverter();
    assert(converter && "Invalid type converter");
    if (!mlir::isa<numba::ntensor::IteratorType>(op.getValue().getType()))
      return mlir::failure();

    auto resType = converter->convertType<mlir::TupleType>(op.getType());
    if (!resType || resType.size() != 2)
      return mlir::failure();

    auto origType =
        mlir::dyn_cast<numba::ntensor::IteratorType>(op.getValue().getType());
    if (!origType)
      return mlir::failure();

    if (origType.getType().getRank() == 0)
      return rewriter.notifyMatchFailure(
          op, "Iteration over 0-rank arrays is not supported");

    auto src = adaptor.getValue();
    auto srcType = mlir::dyn_cast<mlir::TupleType>(src.getType());
    if (!srcType || srcType.size() != 2)
      return mlir::failure();

    auto loc = op.getLoc();
    auto getItem = [&](unsigned i) -> mlir::Value {
      auto idx = rewriter.create<mlir::arith::ConstantIndexOp>(loc, i);
      return rewriter.create<numba::util::TupleExtractOp>(
          loc, srcType.getType(i), src, idx);
    };

    using NTensor = numba::ntensor::NTensorType;
    auto iter = getItem(0);
    auto array = mlir::cast<mlir::TypedValue<NTensor>>(getItem(1));
    auto arrayType = array.getType();

    mlir::Value end = rewriter.create<numba::ntensor::DimOp>(loc, array, 0);
    mlir::Value current = rewriter.create<mlir::memref::LoadOp>(
        loc, iter, /*indices*/ std::nullopt);
    mlir::Value one = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
    mlir::Value next = rewriter.create<mlir::arith::AddIOp>(loc, current, one);

    mlir::Value cond = rewriter.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::slt, current, end);

    rewriter.create<mlir::memref::StoreOp>(loc, next, iter,
                                           /*indices*/ std::nullopt);

    bool slice = arrayType.getRank() > 1;
    auto retType = [&]() -> mlir::Type {
      auto type = mlir::cast<mlir::ShapedType>(array.getType());
      if (!slice)
        return type.getElementType();

      return mlir::cast<NTensor>(resType.getType(0));
    }();

    auto thenBuilder = [&](mlir::OpBuilder &builder, mlir::Location loc) {
      mlir::Value res = [&]() -> mlir::Value {
        if (!slice)
          return builder.create<numba::ntensor::LoadOp>(loc, array, current);

        auto zero = builder.getIndexAttr(0);
        auto one = builder.getIndexAttr(1);

        auto rank = arrayType.getShape().size();
        llvm::SmallVector<mlir::OpFoldResult> offsets(rank, zero);
        offsets[0] = current;

        llvm::SmallVector<mlir::OpFoldResult> sizes(rank);
        for (auto i : llvm::seq<size_t>(0, rank)) {
          mlir::OpFoldResult sz;
          if (i == 0) {
            sz = one;
          } else {
            sz = builder.create<numba::ntensor::DimOp>(loc, array, i)
                     .getResult();
          }
          sizes[i] = sz;
        }

        llvm::SmallVector<mlir::OpFoldResult> strides(rank, one);

        auto subviewType =
            numba::ntensor::SubviewOp::inferRankReducedResultType(
                mlir::cast<NTensor>(retType).getShape(),
                mlir::cast<NTensor>(array.getType()), offsets, sizes, strides);
        mlir::Value arr = builder.create<numba::ntensor::SubviewOp>(
            loc, subviewType, array, offsets, sizes, strides);
        if (arr.getType() != retType)
          arr = builder.create<numba::ntensor::CastOp>(loc, retType, arr);

        return arr;
      }();
      builder.create<mlir::scf::YieldOp>(loc, res);
    };

    auto elseBuilder = [&](mlir::OpBuilder &builder, mlir::Location loc) {
      mlir::Value res =
          builder.create<mlir::ub::PoisonOp>(loc, retType, nullptr);
      builder.create<mlir::scf::YieldOp>(loc, res);
    };

    auto ifOp =
        rewriter.create<mlir::scf::IfOp>(loc, cond, thenBuilder, elseBuilder);

    mlir::Value rets[] = {ifOp.getResult(0), cond};
    mlir::Value ret =
        rewriter.create<numba::util::BuildTupleOp>(loc, resType, rets);
    rewriter.replaceOp(op, ret);
    return mlir::success();
  }
};

template <typename T>
static std::optional<llvm::SmallVector<mlir::Value>>
getElementsValuesIntImpl(mlir::DenseElementsAttr attr, mlir::Location loc,
                         mlir::OpBuilder &builder) {
  auto values = attr.tryGetValues<T>();
  if (mlir::failed(values))
    return std::nullopt;

  llvm::SmallVector<mlir::Value> ret(attr.size());
  auto origElemType = attr.getType().getElementType();
  auto elemType = numba::makeSignlessType(origElemType);
  for (auto &&[i, val] : llvm::enumerate(*values)) {
    mlir::Value res =
        builder.create<mlir::arith::ConstantIntOp>(loc, val, elemType);
    if (origElemType != elemType)
      res = builder.create<numba::util::SignCastOp>(loc, origElemType, res);

    ret[i] = res;
  }

  return ret;
}

template <typename T>
static std::optional<llvm::SmallVector<mlir::Value>>
getElementsValuesFloatImpl(mlir::DenseElementsAttr attr, mlir::Location loc,
                           mlir::OpBuilder &builder) {
  auto values = attr.tryGetValues<T>();
  if (mlir::failed(values))
    return std::nullopt;

  llvm::SmallVector<mlir::Value> ret(attr.size());
  auto elemType =
      mlir::dyn_cast<mlir::FloatType>(attr.getType().getElementType());
  if (!elemType)
    return std::nullopt;

  for (auto &&[i, val] : llvm::enumerate(*values))
    ret[i] = builder.create<mlir::arith::ConstantFloatOp>(
        loc, llvm::APFloat(val), elemType);

  return ret;
}

template <typename T>
static std::optional<llvm::SmallVector<mlir::Value>>
getElementsValuesComplexImpl(mlir::DenseElementsAttr attr, mlir::Location loc,
                             mlir::OpBuilder &builder) {
  auto count = static_cast<unsigned>(attr.size());
  llvm::SmallVector<mlir::Value> ret(count);

  auto elemType = mlir::dyn_cast<mlir::ComplexType>(attr.getElementType());
  if (!elemType)
    return std::nullopt;

  auto complexElemType =
      mlir::dyn_cast<mlir::FloatType>(elemType.getElementType());
  if (!complexElemType || complexElemType.getWidth() != (sizeof(T) * 8))
    return std::nullopt;

  auto ptr = attr.getRawData().data();
  using CType = std::complex<T>;
  auto stride = attr.isSplat() ? 0 : sizeof(CType);
  for (auto i : llvm::seq(0u, count)) {
    auto &val = *reinterpret_cast<const CType *>(ptr + stride * i);
    const mlir::Attribute vals[] = {
        mlir::FloatAttr::get(complexElemType, val.real()),
        mlir::FloatAttr::get(complexElemType, val.imag()),
    };
    auto arr = builder.getArrayAttr(vals);
    ret[i] = builder.create<mlir::complex::ConstantOp>(loc, elemType, arr);
  }

  return ret;
}

static std::optional<llvm::SmallVector<mlir::Value>>
getElementsValues(mlir::DenseElementsAttr attr, mlir::Location loc,
                  mlir::OpBuilder &builder) {
  using func_t = std::optional<llvm::SmallVector<mlir::Value>> (*)(
      mlir::DenseElementsAttr, mlir::Location, mlir::OpBuilder &);
  const func_t funcs[] = {
      // clang-format off
    &getElementsValuesIntImpl<int8_t>,
    &getElementsValuesIntImpl<int16_t>,
    &getElementsValuesIntImpl<int32_t>,
    &getElementsValuesIntImpl<int64_t>,
    &getElementsValuesFloatImpl<float>,
    &getElementsValuesFloatImpl<double>,
    &getElementsValuesComplexImpl<float>,
    &getElementsValuesComplexImpl<double>,
      // clang-format on
  };

  for (auto func : funcs)
    if (auto res = func(attr, loc, builder))
      return res;

  return std::nullopt;
}

struct LowerConst : public mlir::OpConversionPattern<plier::ConstOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::ConstOp op, plier::ConstOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    assert(converter && "Invalid type converter");
    auto resType = mlir::dyn_cast_or_null<numba::ntensor::NTensorType>(
        converter->convertType(op.getType()));
    if (!resType)
      return rewriter.notifyMatchFailure(op, [&](mlir::Diagnostic &diag) {
        diag << "Invalid result type " << op.getType();
      });

    auto attr =
        mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(op.getValAttr());
    if (!attr)
      return rewriter.notifyMatchFailure(op, [&](mlir::Diagnostic &diag) {
        diag << "Invalid value type " << op.getValAttr();
      });

    auto constType = attr.getType();
    if (!constType.hasStaticShape())
      if (!attr)
        return rewriter.notifyMatchFailure(op, [&](mlir::Diagnostic &diag) {
          diag << "Expected a static shape but got " << constType;
        });

    auto loc = op.getLoc();
    auto values = getElementsValues(attr, loc, rewriter);
    if (!values)
      return rewriter.notifyMatchFailure(op, [&](mlir::Diagnostic &diag) {
        diag << "Failed to extract values from " << attr;
      });

    auto elemType = constType.getElementType();
    auto constTensorType =
        numba::ntensor::NTensorType::get(constType.getShape(), elemType);

    mlir::Value res = rewriter.create<numba::ntensor::FromElementsOp>(
        loc, constTensorType, *values);
    if (constTensorType != resType)
      res = rewriter.create<numba::ntensor::CastOp>(loc, resType, res);

    rewriter.replaceOp(op, res);
    return mlir::success();
  }
};

static bool isNtensor(mlir::TypeConverter &converter, mlir::Type type) {
  return !!converter.convertType(type)
               .dyn_cast_or_null<numba::ntensor::NTensorType>();
}

struct PlierToNtensorPass
    : public mlir::PassWrapper<PlierToNtensorPass, mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PlierToNtensorPass)

  PlierToNtensorPass()
      : resolver(std::make_shared<NumpyResolver>("numba_mlir.mlir.numpy.funcs",
                                                 "_get_func")) {}

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<numba::ntensor::NTensorDialect>();
    registry.insert<numba::util::NumbaUtilDialect>();
    registry.insert<plier::PlierDialect>();
  }

  void runOnOperation() override {
    auto &context = getContext();

    mlir::TypeConverter typeConverter;
    // Convert unknown types to itself
    typeConverter.addConversion([](mlir::Type type) { return type; });

    numba::populateTupleTypeConverter(typeConverter);
    typeConverter.addConversion(
        [](plier::SliceType type) -> std::optional<mlir::Type> {
          return numba::ntensor::SliceType::get(type.getContext());
        });

    auto indexType = mlir::IndexType::get(&context);
    auto indexMemref = mlir::MemRefType::get({}, indexType);
    typeConverter.addConversion(
        [indexMemref](numba::ntensor::IteratorType type) {
          return mlir::TupleType::get(type.getContext(),
                                      {indexMemref, type.getType()});
        });

    auto addUnrealizedCast = [](mlir::OpBuilder &builder, mlir::Type type,
                                mlir::ValueRange inputs, mlir::Location loc) {
      auto cast =
          builder.create<mlir::UnrealizedConversionCastOp>(loc, type, inputs);
      return std::optional<mlir::Value>(cast.getResult(0));
    };
    typeConverter.addArgumentMaterialization(addUnrealizedCast);
    typeConverter.addSourceMaterialization(addUnrealizedCast);
    typeConverter.addTargetMaterialization(addUnrealizedCast);

    mlir::RewritePatternSet patterns(&context);
    mlir::ConversionTarget target(context);

    numba::populateControlFlowTypeConversionRewritesAndTarget(typeConverter,
                                                              patterns, target);
    numba::populateTupleTypeConversionRewritesAndTarget(typeConverter, patterns,
                                                        target);

    target.addDynamicallyLegalOp<plier::GetItemOp>(
        [&typeConverter](plier::GetItemOp op) -> std::optional<bool> {
          auto containerType = op.getValue().getType();
          if (isNtensor(typeConverter, containerType))
            return false;

          return std::nullopt;
        });
    target.addDynamicallyLegalOp<plier::SetItemOp>(
        [&typeConverter](plier::SetItemOp op) -> std::optional<bool> {
          auto containerType = op.getTarget().getType();
          if (isNtensor(typeConverter, containerType))
            return false;

          return std::nullopt;
        });

    target.addDynamicallyLegalOp<numba::ntensor::GetitemOp,
                                 numba::ntensor::SetitemOp>(
        [&typeConverter](mlir::Operation *op) {
          return typeConverter.isLegal(op);
        });

    target.addDynamicallyLegalOp<plier::UnaryOp>(
        [&typeConverter](plier::UnaryOp op) -> std::optional<bool> {
          auto val = op.getValue().getType();
          if (isNtensor(typeConverter, val))
            return false;

          return std::nullopt;
        });

    target.addDynamicallyLegalOp<plier::BinOp>(
        [&typeConverter](plier::BinOp op) -> std::optional<bool> {
          auto lhs = op.getLhs().getType();
          auto rhs = op.getRhs().getType();
          if (isNtensor(typeConverter, lhs) || isNtensor(typeConverter, rhs))
            return false;

          return std::nullopt;
        });
    target.addDynamicallyLegalOp<plier::InplaceBinOp>(
        [&typeConverter](plier::InplaceBinOp op) -> std::optional<bool> {
          auto lhs = op.getLhs().getType();
          auto rhs = op.getRhs().getType();
          if (isNtensor(typeConverter, lhs) || isNtensor(typeConverter, rhs))
            return false;

          return std::nullopt;
        });

    target.addDynamicallyLegalOp<plier::PyCallOp>(
        [this](plier::PyCallOp op) -> std::optional<bool> {
          auto funcName = op.getFuncName();
          if (resolver->hasFunc(funcName))
            return false;

          return std::nullopt;
        });

    target.addDynamicallyLegalOp<plier::GetattrOp>(
        [&typeConverter](plier::GetattrOp op) -> std::optional<bool> {
          auto containerType = op.getValue().getType();
          if (isNtensor(typeConverter, containerType) &&
              !mlir::isa<plier::BoundFunctionType>(op.getResult().getType()))
            return false;

          return std::nullopt;
        });

    target.addDynamicallyLegalOp<plier::CastOp>(
        [&typeConverter](plier::CastOp op) -> std::optional<bool> {
          auto srcType = op.getValue().getType();
          auto dstType = op.getType();
          if (isNtensor(typeConverter, srcType) ||
              isNtensor(typeConverter, dstType))
            return false;

          return true;
        });

    target.addDynamicallyLegalOp<numba::util::TupleExtractOp>(
        [](numba::util::TupleExtractOp op) -> std::optional<bool> {
          if (auto elemType = isUniTuple(op.getSource().getType()))
            if (numba::ntensor::NTensorType::isValidElementType(*elemType))
              return false;

          return std::nullopt;
        });

    target.addDynamicallyLegalOp<plier::ConstOp>(
        [&typeConverter](plier::ConstOp op) {
          return !isNtensor(typeConverter, op.getType());
        });

    target.addDynamicallyLegalOp<plier::GetiterOp>(
        [&](plier::GetiterOp op) -> std::optional<bool> {
          return !mlir::isa<numba::ntensor::NTensorType>(
              op.getValue().getType());
        });
    target.addDynamicallyLegalOp<plier::IternextOp>(
        [&](plier::IternextOp op) -> std::optional<bool> {
          return !mlir::isa<numba::ntensor::IteratorType>(
              op.getValue().getType());
        });

    target.addIllegalOp<plier::BuildSliceOp>();

    target.addLegalDialect<numba::ntensor::NTensorDialect>();

    patterns.insert<
        // clang-format off
        GetitemToNtensor,
        SetitemToNtensor,
        NtensorGetitemToNtensor,
        NtensorSetitemToNtensor,
        UnaryToNtensor,
        BinopToNtensor,
        InplaceBinopToNtensor,
        BuildSliceToNtensor,
        CastsToNtensor,
        UnitupleExtractToNtensor,
        GetitertConversionPattern,
        IternextConversionPattern,
        LowerConst
        // clang-format on
        >(typeConverter, &context);

    patterns.insert<NumpyCallsToNtensor, NumpyAttrsToNtensor>(
        typeConverter, &context, *resolver);

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns))))
      signalPassFailure();
  }

private:
  std::shared_ptr<NumpyResolver> resolver;
};

struct GetitemArrayOpLowering
    : public mlir::OpRewritePattern<numba::ntensor::GetitemOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(numba::ntensor::GetitemOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto src = op.getSource();
    if (!src.getType().isa<numba::ntensor::NTensorType>())
      return mlir::failure();

    auto index = op.getIndex();
    if (!index.getType().isa<numba::ntensor::NTensorType>())
      return mlir::failure();

    mlir::StringRef opName = "array.__getitem__";
    auto resType = op.getType();
    mlir::Value args[] = {src, index};
    rewriter.replaceOpWithNewOp<numba::ntensor::PrimitiveOp>(op, resType, args,
                                                             opName);
    return mlir::success();
  }
};

struct SetitemArrayOpLowering
    : public mlir::OpRewritePattern<numba::ntensor::SetitemOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(numba::ntensor::SetitemOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto src = op.getSource();
    auto srcType = src.getType().dyn_cast<numba::ntensor::NTensorType>();
    if (!srcType || srcType.getRank() != 1)
      return mlir::failure();

    auto index = op.getIndex();
    auto indexType = index.getType().dyn_cast<numba::ntensor::NTensorType>();
    if (!indexType || indexType.getRank() != 1)
      return mlir::failure();

    auto val = op.getValue();
    auto valueType = val.getType().dyn_cast<numba::ntensor::NTensorType>();
    if (!valueType || valueType.getRank() != 1)
      return mlir::failure();

    mlir::StringRef opName = "array.__setitem__";
    mlir::Value args[] = {src, index, val};

    auto loc = op.getLoc();
    auto res =
        rewriter.create<numba::ntensor::PrimitiveOp>(loc, srcType, args, opName)
            .getResult(0);
    rewriter.create<numba::ntensor::CopyOp>(loc, res, src);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

static PyLinalgResolver::Values
castRetTypes(mlir::Location loc, mlir::PatternRewriter &rewriter,
             mlir::TypeRange resultTypes,
             std::optional<PyLinalgResolver::Values> vals) {
  auto results = std::move(vals).value();
  assert(results.size() == resultTypes.size());
  for (auto &&[i, ret] : llvm::enumerate(results)) {
    auto dstType = resultTypes[i];

    auto srcType = ret.getType();
    if (dstType != srcType)
      results[i] = doSafeCast(rewriter, loc, ret, dstType);
  }
  return results;
}

static mlir::FailureOr<mlir::Attribute> getEnvAttr(mlir::Operation *op) {
  assert(op && "Invalid op");

  mlir::Attribute env;
  for (auto types : {mlir::TypeRange(op->getOperandTypes()),
                     mlir::TypeRange(op->getResultTypes())}) {
    for (auto type : types) {
      auto tensor = type.dyn_cast<numba::ntensor::NTensorType>();
      if (!tensor)
        continue;

      if (!env) {
        env = tensor.getEnvironment();
      } else if (env != tensor.getEnvironment()) {
        return mlir::failure();
      }
    }
  }

  return env;
}

static mlir::FailureOr<PyLinalgResolver::Values>
rewritePrimitiveFunc(mlir::PatternRewriter &rewriter, mlir::Location loc,
                     const PyLinalgResolver &resolver, mlir::ValueRange args,
                     mlir::TypeRange resultTypes, mlir::Attribute env,
                     llvm::StringRef opName) {
  auto getRes = [&]() -> std::optional<PyLinalgResolver::Values> {
    auto funcRes =
        resolver.rewriteFunc(opName, loc, rewriter, args, std::nullopt);
    if (funcRes)
      return funcRes;

    auto isNone = [](mlir::Value val) {
      return mlir::isa<mlir::NoneType>(val.getType());
    };

    if (opName.startswith("array.") && args.size() >= 1 &&
        llvm::all_of(args.drop_front(), isNone))
      return resolver.rewriteAttr(opName, loc, rewriter, args.front());

    return std::nullopt;
  };

  PyLinalgResolver::Values newRes;
  if (env != nullptr) {
    auto regionOp = rewriter.create<numba::util::EnvironmentRegionOp>(
        loc, env, /*args*/ std::nullopt, resultTypes);
    auto &newBody = regionOp.getRegion().front();
    rewriter.eraseOp(newBody.getTerminator());

    mlir::OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(&newBody);
    auto res = getRes();
    if (!res) {
      rewriter.eraseOp(regionOp);
      return mlir::failure();
    }

    auto results = castRetTypes(loc, rewriter, resultTypes, *res);
    rewriter.create<numba::util::EnvironmentRegionYieldOp>(loc, results);

    auto regResults = regionOp.getResults();
    newRes.assign(regResults.begin(), regResults.end());
  } else {
    auto res = getRes();
    if (!res)
      return mlir::failure();

    auto results = castRetTypes(loc, rewriter, resultTypes, *res);
    newRes.assign(results.begin(), results.end());
  }

  return newRes;
}

struct NtensorPrimitiveCallsLowering final
    : public mlir::OpRewritePattern<numba::ntensor::PrimitiveOp> {
  NtensorPrimitiveCallsLowering(mlir::MLIRContext *context)
      : OpRewritePattern(context),
        resolver("numba_mlir.mlir.numpy.funcs", "registry") {}

  mlir::LogicalResult
  matchAndRewrite(numba::ntensor::PrimitiveOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto env = getEnvAttr(op);
    if (mlir::failed(env))
      return mlir::failure();

    auto opName = op.getOp();
    auto loc = op.getLoc();
    auto newRes = rewritePrimitiveFunc(rewriter, loc, resolver, op.getArgs(),
                                       op.getResultTypes(), *env, opName);
    if (mlir::failed(newRes))
      return mlir::failure();

    rerunScfPipeline(op);
    rewriter.replaceOp(op, *newRes);
    return mlir::success();
  }

private:
  PyLinalgResolver resolver;
};

struct NtensorPrimitiveSeCallsLowering final
    : public mlir::OpRewritePattern<numba::ntensor::PrimitiveSeOp> {
  NtensorPrimitiveSeCallsLowering(mlir::MLIRContext *context)
      : OpRewritePattern(context),
        resolver("numba_mlir.mlir.numpy.funcs", "registry") {}

  mlir::LogicalResult
  matchAndRewrite(numba::ntensor::PrimitiveSeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto env = getEnvAttr(op);
    if (mlir::failed(env))
      return mlir::failure();

    auto opName = op.getOp();
    auto loc = op.getLoc();
    auto newRes = rewritePrimitiveFunc(rewriter, loc, resolver, op.getArgs(),
                                       op.getResultTypes(), *env, opName);
    if (mlir::failed(newRes))
      return mlir::failure();

    rerunScfPipeline(op);
    rewriter.replaceOp(op, *newRes);
    return mlir::success();
  }

private:
  PyLinalgResolver resolver;
};

struct NtensorViewPrimitiveCallsLowering final
    : public mlir::OpRewritePattern<numba::ntensor::ViewPrimitiveOp> {
  NtensorViewPrimitiveCallsLowering(mlir::MLIRContext *context)
      : OpRewritePattern(context),
        resolver("numba_mlir.mlir.numpy.funcs", "registry") {}

  mlir::LogicalResult
  matchAndRewrite(numba::ntensor::ViewPrimitiveOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto env = getEnvAttr(op);
    if (mlir::failed(env))
      return mlir::failure();

    auto opName = op.getOp();
    auto loc = op.getLoc();
    auto newRes =
        rewritePrimitiveFunc(rewriter, loc, resolver, op.getOperands(),
                             op->getResultTypes(), *env, opName);
    if (mlir::failed(newRes))
      return mlir::failure();

    rerunScfPipeline(op);
    rewriter.replaceOp(op, *newRes);
    return mlir::success();
  }

private:
  PyLinalgResolver resolver;
};

struct NumpyCallsResolver
    : public mlir::OpRewritePattern<numba::ntensor::CallOp> {
  NumpyCallsResolver(mlir::MLIRContext *ctx, NumpyResolver &r)
      : OpRewritePattern(ctx), resolver(r) {}

  mlir::LogicalResult
  matchAndRewrite(numba::ntensor::CallOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto funcName = op.getOp();

    auto loc = op.getLoc();
    llvm::SmallVector<mlir::Value> args;
    llvm::SmallVector<mlir::Value> outResults;
    PrimitiveType primitive_type = PrimitiveType::Default;
    if (mlir::failed(resolver.resolveFuncArgs(
            rewriter, loc, funcName, op.getArgs(), op.getArgsNames(), args,
            outResults, primitive_type)))
      return mlir::failure();

    if (primitive_type == PrimitiveType::View) {
      if (op.getNumResults() != 1 || args.size() < 1)
        return mlir::failure();

      if (!op.getResult(0).getType().isa<numba::ntensor::NTensorType>() ||
          !args[0].getType().isa<numba::ntensor::NTensorType>())
        return mlir::failure();
    }

    auto results = [&]() -> mlir::ValueRange {
      switch (primitive_type) {
      case PrimitiveType::Default:
        return rewriter
            .create<numba::ntensor::PrimitiveOp>(loc, op->getResultTypes(),
                                                 args, funcName)
            .getResults();
      case PrimitiveType::View:
        return rewriter
            .create<numba::ntensor::ViewPrimitiveOp>(
                loc, op.getResult(0).getType(), args.front(),
                llvm::ArrayRef(args).drop_front(), funcName)
            ->getResults();
      case PrimitiveType::SideEffect:
        return rewriter
            .create<numba::ntensor::PrimitiveSeOp>(loc, op->getResultTypes(),
                                                   args, funcName)
            .getResults();
      }
      std::string err =
          "Invalid primitive type " + std::to_string(int(primitive_type));
      llvm_unreachable(err.c_str());
    }();

    if (primitive_type != PrimitiveType::SideEffect)
      for (auto &&[dst, src] : llvm::zip(outResults, results))
        rewriter.create<numba::ntensor::CopyOp>(loc, src, dst);

    rewriter.replaceOp(op, results);
    return mlir::success();
  }

private:
  NumpyResolver &resolver;
};

struct UnaryOpsLowering
    : public mlir::OpRewritePattern<numba::ntensor::UnaryOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(numba::ntensor::UnaryOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto opName = op.getOp();
    const std::pair<llvm::StringRef, llvm::StringRef> mapping[] = {
        {"-", "neg"},
        {"+", "pos"},
        {"~", "invert"},
        {"not", "not"},
    };

    for (auto &&[srcName, dstName] : mapping) {
      if (opName != srcName)
        continue;

      llvm::SmallVector<char> tmp;
      auto newName = ("operator." + dstName).toStringRef(tmp);
      rewriter.replaceOpWithNewOp<numba::ntensor::PrimitiveOp>(
          op, op->getResultTypes(), op.getValue(), newName);
      return mlir::success();
    }

    return mlir::failure();
  }
};

struct BinOpsLowering
    : public mlir::OpRewritePattern<numba::ntensor::BinaryOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(numba::ntensor::BinaryOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto name = op.getOp();
    for (auto it : plier::getOperators()) {
      if (it.op == name) {
        auto newName = (llvm::Twine("operator.") + it.name).str();
        mlir::Value args[] = {op.getLhs(), op.getRhs()};
        rewriter.replaceOpWithNewOp<numba::ntensor::PrimitiveOp>(
            op, op->getResultTypes(), args, newName);
        return mlir::success();
      }
    }
    return mlir::failure();
  }
};

struct ResolveNtensorPass
    : public mlir::PassWrapper<ResolveNtensorPass, mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ResolveNtensorPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::bufferization::BufferizationDialect>();
    registry.insert<mlir::linalg::LinalgDialect>();
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<mlir::tensor::TensorDialect>();
    registry.insert<mlir::ub::UBDialect>();
    registry.insert<numba::ntensor::NTensorDialect>();
    registry.insert<numba::util::NumbaUtilDialect>();
  }

  void runOnOperation() override {
    auto &ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);

    patterns
        .insert<NtensorPrimitiveCallsLowering, NtensorPrimitiveSeCallsLowering,
                NtensorViewPrimitiveCallsLowering>(&ctx);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                        std::move(patterns))))
      return signalPassFailure();
  }
};

struct WrapParforRegionsPass
    : public mlir::PassWrapper<WrapParforRegionsPass,
                               mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(WrapParforRegionsPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::scf::SCFDialect>();
    registry.insert<numba::ntensor::NTensorDialect>();
    registry.insert<numba::util::NumbaUtilDialect>();
  }

  void runOnOperation() override {
    auto op = getOperation();

    auto getOpEnv = [](mlir::Operation *op) -> std::optional<mlir::Attribute> {
      if (auto load = mlir::dyn_cast<numba::ntensor::LoadOp>(op))
        return load.getArray().getType().getEnvironment();

      if (auto store = mlir::dyn_cast<numba::ntensor::StoreOp>(op))
        return store.getArray().getType().getEnvironment();

      return std::nullopt;
    };

    mlir::OpBuilder builder(&getContext());
    llvm::SmallVector<std::pair<mlir::scf::ForOp, mlir::Attribute>>
        opsToProcess;

    auto visitor = [&](mlir::scf::ForOp forOp) -> mlir::WalkResult {
      if (!isInsideParallelRegion(forOp))
        return mlir::WalkResult::advance();

      std::optional<mlir::Attribute> env;
      auto innerVisitor = [&](mlir::Operation *innerOp) -> mlir::WalkResult {
        auto opEnv = getOpEnv(innerOp);
        if (!opEnv || !*opEnv)
          return mlir::WalkResult::advance();

        if (!env) {
          env = *opEnv;
          return mlir::WalkResult::advance();
        }
        auto res = numba::util::mergeEnvAttrs(*env, *opEnv);
        if (!res) {
          forOp->emitError("Incompatible envs: ") << *env << " and " << *opEnv;
          return mlir::WalkResult::interrupt();
        }
        env = *res;
        return mlir::WalkResult::advance();
      };
      if (forOp->walk(innerVisitor).wasInterrupted())
        return mlir::WalkResult::interrupt();

      if (env && *env)
        opsToProcess.emplace_back(forOp, *env);

      return mlir::WalkResult::advance();
    };
    if (op->walk(visitor).wasInterrupted())
      return signalPassFailure();

    if (opsToProcess.empty())
      return markAllAnalysesPreserved();

    for (auto &&[forOp, env] : opsToProcess) {
      auto resultTypes = forOp.getResultTypes();
      builder.setInsertionPoint(forOp);
      auto envRegion = builder.create<numba::util::EnvironmentRegionOp>(
          forOp.getLoc(), env, /*args*/ std::nullopt, resultTypes);
      auto &envRegionBlock = envRegion.getRegion().front();
      auto term = envRegionBlock.getTerminator();
      forOp->moveBefore(term);
      forOp->replaceAllUsesWith(envRegion.getResults());
      builder.setInsertionPoint(term);
      builder.create<numba::util::EnvironmentRegionYieldOp>(term->getLoc(),
                                                            forOp.getResults());
      term->erase();
    }
  }
};

struct MarkInputShapesRanges
    : public mlir::PassWrapper<MarkInputShapesRanges,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MarkInputShapesRanges)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<numba::util::NumbaUtilDialect>();
  }

  void runOnOperation() override {
    auto mod = getOperation();

    auto getRange = [&](int64_t min, int64_t max) {
      return numba::util::IndexRangeAttr::get(&getContext(), min, max);
    };

    auto attrName = mlir::StringAttr::get(
        mod.getContext(), numba::util::attributes::getShapeRangeName());
    mod.walk([&](mlir::FunctionOpInterface func) {
      if (func.isExternal())
        return;

      auto &body = func.getFunctionBody();
      assert(!body.empty());
      for (auto &&[i, arg] : llvm::enumerate(body.front().getArguments())) {
        auto shaped = arg.getType().dyn_cast<mlir::ShapedType>();
        if (!shaped)
          continue;

        auto newRange = [&]() -> std::optional<mlir::ArrayAttr> {
          auto uses = mlir::SymbolTable::getSymbolUses(func, mod);
          if (!uses || !uses->empty())
            return std::nullopt;

          auto rank = static_cast<unsigned>(shaped.getRank());
          llvm::SmallVector<mlir::Attribute> shapeRanges(rank);

          for (auto &&[i, dim] : llvm::enumerate(shaped.getShape())) {
            if (mlir::ShapedType::isDynamic(dim)) {
              shapeRanges[i] = getRange(2, std::numeric_limits<int64_t>::max());
            } else {
              shapeRanges[i] = getRange(dim, dim);
            }
          }

          return mlir::ArrayAttr::get(&getContext(), shapeRanges);
        }();

        if (!newRange)
          continue;

        func.setArgAttr(static_cast<unsigned>(i), attrName, *newRange);
      }
    });
  }
};

struct PropagateFastmathFlags
    : public mlir::PassWrapper<PropagateFastmathFlags,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PropagateFastmathFlags)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<numba::util::NumbaUtilDialect>();
  }

  void runOnOperation() override {
    auto func = getOperation();
    if (!func->hasAttr(numba::util::attributes::getFastmathName()))
      return markAllAnalysesPreserved();

    auto newFmf = mlir::arith::FastMathFlagsAttr::get(
        &getContext(), mlir::arith::FastMathFlags::fast);
    auto visitor = [&](mlir::arith::ArithFastMathInterface fmi) {
      if (fmi.getFastMathFlagsAttr() == newFmf)
        return;

      auto attrName = fmi.getFastMathAttrName();
      fmi->setAttr(attrName, newFmf);
    };
    func->walk(visitor);
  }
};

struct ResolveNumpyFuncsPass
    : public mlir::PassWrapper<ResolveNumpyFuncsPass,
                               mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ResolveNumpyFuncsPass)

  ResolveNumpyFuncsPass()
      : resolver(std::make_shared<NumpyResolver>("numba_mlir.mlir.numpy.funcs",
                                                 "_get_func")) {}

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<numba::ntensor::NTensorDialect>();
    registry.insert<numba::util::NumbaUtilDialect>();
  }

  void runOnOperation() override {
    auto &ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);

    numba::ntensor::populateResolveArrayOpsPatterns(patterns);

    patterns.insert<NumpyCallsResolver>(&ctx, *resolver);

    patterns.insert<GetitemArrayOpLowering, SetitemArrayOpLowering,
                    UnaryOpsLowering, BinOpsLowering>(&ctx);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                        std::move(patterns))))
      return signalPassFailure();
  }

private:
  std::shared_ptr<NumpyResolver> resolver;
};

struct SimplifyExpandDims
    : public mlir::OpRewritePattern<mlir::linalg::GenericOp> {
  using mlir::OpRewritePattern<mlir::linalg::GenericOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::linalg::GenericOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics())
      return mlir::failure();

    if (op.getInputs().size() != 1 || op.getOutputs().size() != 1)
      return mlir::failure();

    auto context = op.getContext();
    auto parallelAttr = mlir::linalg::IteratorTypeAttr::get(
        context, mlir::utils::IteratorType::parallel);

    if (llvm::any_of(op.getIteratorTypes(),
                     [&](auto attr) { return attr != parallelAttr; }))
      return mlir::failure();

    auto maps = op.getIndexingMaps();
    assert(maps.size() == 2);
    auto outMap = maps[1].cast<mlir::AffineMapAttr>().getValue();
    if (!outMap.isIdentity())
      return mlir::failure();

    auto inMap = maps[0].cast<mlir::AffineMapAttr>().getValue();
    auto numDims = op.getNumLoops();
    if (inMap.getNumResults() != numDims)
      return mlir::failure();

    bool changed = false;
    auto outShape =
        op.getOutputs()[0].getType().cast<mlir::RankedTensorType>().getShape();
    llvm::SmallVector<mlir::AffineExpr> exprs(numDims);
    for (unsigned i = 0; i < numDims; ++i) {
      auto prevExpr = inMap.getResult(i);
      bool canConvert = [&]() {
        if (outShape[i] == 1) {
          auto constExpr = mlir::dyn_cast<mlir::AffineConstantExpr>(prevExpr);
          if (constExpr && constExpr.getValue() == 0)
            return true;
        }
        return false;
      }();
      if (canConvert) {
        changed = true;
        exprs[i] = mlir::getAffineDimExpr(i, context);
      } else {
        exprs[i] = prevExpr;
      }
    }

    if (changed) {
      const mlir::Attribute newMaps[] = {
          mlir::AffineMapAttr::get(
              mlir::AffineMap::get(numDims, 0, exprs, context)),
          maps[1]};
      auto newMapsAttr = mlir::ArrayAttr::get(context, newMaps);
      rewriter.modifyOpInPlace(op,
                               [&]() { op.setIndexingMapsAttr(newMapsAttr); });
    }

    return mlir::success(changed);
  }
};

struct LowerEnforceShape
    : public mlir::OpRewritePattern<numba::util::EnforceShapeOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(numba::util::EnforceShapeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto type = op.getType();
    auto src = op.getValue();
    rewriter.replaceOpWithNewOp<mlir::tensor::CastOp>(op, type, src);
    return mlir::success();
  }
};

struct InsertSliceToPad
    : public mlir::OpRewritePattern<mlir::tensor::InsertSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::tensor::InsertSliceOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto gen = op.getDest().getDefiningOp<mlir::tensor::GenerateOp>();
    if (!gen)
      return mlir::failure();

    for (auto stride : op.getMixedStrides()) {
      auto val = mlir::getConstantIntValue(stride);
      if (!val || *val != 1)
        return mlir::failure();
    }

    auto src = op.getSource();
    auto srcType = src.getType().cast<mlir::RankedTensorType>();
    auto dstType = gen.getType().cast<mlir::RankedTensorType>();

    auto rank = static_cast<unsigned>(srcType.getRank());

    auto low = op.getMixedOffsets();
    llvm::SmallVector<mlir::OpFoldResult> high(rank);

    auto loc = op->getLoc();

    auto toVal = [&](mlir::OpFoldResult val) -> mlir::Value {
      if (val.is<mlir::Value>())
        return val.get<mlir::Value>();

      return rewriter.create<mlir::arith::ConstantOp>(
          loc, mlir::cast<mlir::TypedAttr>(val.get<mlir::Attribute>()));
    };

    for (auto i : llvm::seq(0u, rank)) {
      auto dstDim = rewriter.createOrFold<mlir::tensor::DimOp>(loc, gen, i);
      auto srcDim = rewriter.createOrFold<mlir::tensor::DimOp>(loc, src, i);
      auto offset = rewriter.createOrFold<mlir::arith::AddIOp>(
          loc, toVal(srcDim), toVal(low[i]));
      offset = rewriter.createOrFold<mlir::arith::SubIOp>(loc, toVal(dstDim),
                                                          offset);
      high[i] = mlir::getAsOpFoldResult(offset);
    }

    auto pad =
        rewriter.create<mlir::tensor::PadOp>(loc, dstType, src, low, high);
    rewriter.cloneRegionBefore(gen.getRegion(), pad.getRegion(),
                               pad.getRegion().end());
    rewriter.replaceOp(op, pad.getResult());
    return mlir::success();
  }
};

struct GenerateToFill
    : public mlir::OpRewritePattern<mlir::tensor::GenerateOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::tensor::GenerateOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto &body = op.getBody();
    if (!llvm::hasSingleElement(body))
      return mlir::failure();

    auto &block = body.getBlocks().front();
    if (!llvm::hasSingleElement(block))
      return mlir::failure();

    auto term = mlir::cast<mlir::tensor::YieldOp>(block.getTerminator());
    auto resType = op.getType().cast<mlir::ShapedType>();

    auto loc = op->getLoc();
    mlir::Value init = rewriter.create<mlir::tensor::EmptyOp>(
        loc, resType.getShape(), resType.getElementType(),
        op.getDynamicExtents());

    rewriter.replaceOpWithNewOp<mlir::linalg::FillOp>(op, term.getValue(),
                                                      init);
    return mlir::success();
  }
};

struct SliceOfGeneric : public mlir::OpRewritePattern<mlir::linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::linalg::GenericOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics())
      return mlir::failure();

    if (op->getNumResults() != 1)
      return mlir::failure();

    auto res = op->getResult(0);
    if (!res.hasOneUse())
      return mlir::failure();

    mlir::Operation *user = *(res.getUsers().begin());
    if (!mlir::isa<mlir::tensor::ExtractSliceOp, mlir::tensor::ExtractOp>(user))
      return mlir::failure();

    auto output = op.getOutputs().front();

    auto resType = res.getType().cast<mlir::RankedTensorType>();
    auto resRank = static_cast<unsigned>(resType.getRank());
    auto maps = [&]() {
      auto mapsList =
          op.getIndexingMaps().getAsValueRange<mlir::AffineMapAttr>();
      return llvm::SmallVector<mlir::AffineMap>(mapsList.begin(),
                                                mapsList.end());
    }();
    assert(!maps.empty());
    for (auto m : maps)
      if (!m.isProjectedPermutation())
        return mlir::failure();

    auto resMap = maps.back();

    auto iters = op.getIteratorTypes();
    auto parallelIter = mlir::linalg::IteratorTypeAttr::get(
        rewriter.getContext(), mlir::utils::IteratorType::parallel);
    for (auto i : llvm::seq(0u, resRank)) {
      auto dim = resMap.getDimPosition(i);
      assert(dim < iters.size());
      if (iters[dim] != parallelIter)
        return mlir::failure();
    }

    bool extractElem = false;
    llvm::SmallBitVector droppedDims;
    llvm::SmallVector<mlir::OpFoldResult, 4> offsets;
    llvm::SmallVector<mlir::OpFoldResult, 4> sizes;
    llvm::SmallVector<mlir::OpFoldResult, 4> strides;

    auto zero = rewriter.getIndexAttr(0);
    auto one = rewriter.getIndexAttr(1);

    auto assignArr = [](llvm::SmallVectorImpl<mlir::OpFoldResult> &arr,
                        const auto &range) {
      arr.reserve(range.size());
      arr.assign(range.begin(), range.end());
    };

    if (auto sliceOp = mlir::dyn_cast<mlir::tensor::ExtractSliceOp>(user)) {
      offsets = sliceOp.getMixedOffsets();
      sizes = sliceOp.getMixedSizes();
      strides = sliceOp.getMixedStrides();
      droppedDims = sliceOp.getDroppedDims();
    } else if (auto extractOp = mlir::dyn_cast<mlir::tensor::ExtractOp>(user)) {
      if (extractOp.getIndices().empty())
        return mlir::failure();

      extractElem = true;
      assignArr(offsets, extractOp.getIndices());
      sizes.resize(offsets.size(), one);
      strides.resize(offsets.size(), one);
      droppedDims.resize(offsets.size(), true);
    } else {
      llvm_unreachable("Invalid op");
    }

    auto oldInputs = op.getInputs();
    llvm::SmallVector<mlir::Value, 4> newInputs(oldInputs.size());

    auto ctx = getContext();
    auto replaceAffineDim = [&](mlir::AffineExpr expr, unsigned srcDim,
                                unsigned dstDim) {
      auto src = mlir::getAffineDimExpr(srcDim, ctx);
      auto dst = mlir::getAffineDimExpr(dstDim, ctx);
      return expr.replace(src, dst);
    };
    auto findResDim = [&](unsigned inputDim) -> std::optional<unsigned> {
      for (auto d : llvm::seq(0u, resRank)) {
        if (resMap.getDimPosition(d) == inputDim)
          return d;
      }
      return std::nullopt;
    };
    auto isDroppedDim = [&](unsigned d) -> bool {
      if (auto indVal = findResDim(d)) {
        auto ind = *indVal;
        assert(ind < droppedDims.size());
        return droppedDims[ind];
      }
      return false;
    };

    auto numLoops = static_cast<unsigned>(iters.size());
    auto ErasedLoop = static_cast<unsigned>(-1);
    llvm::SmallVector<unsigned, 4> loopsMapping(numLoops, ErasedLoop);
    llvm::SmallVector<mlir::utils::IteratorType, 4> newIters;
    newIters.reserve(numLoops);
    for (auto d : llvm::seq(0u, numLoops)) {
      if (!isDroppedDim(d)) {
        auto i = newIters.size();
        assert(i != ErasedLoop);
        newIters.emplace_back(
            iters[d].cast<mlir::linalg::IteratorTypeAttr>().getValue());
        loopsMapping[d] = i;
      }
    }
    auto finalNumLoops = static_cast<unsigned>(newIters.size());

    llvm::SmallVector<mlir::AffineExpr, 4> tempExprs;
    tempExprs.reserve(numLoops);

    auto updateMap = [&](mlir::AffineMap srcMap) -> mlir::AffineMap {
      if (finalNumLoops == numLoops)
        return srcMap;

      tempExprs.clear();
      auto mapResults = srcMap.getResults();
      for (auto i : llvm::seq<size_t>(0, mapResults.size())) {
        auto origLoop = srcMap.getDimPosition(i);
        assert(origLoop < loopsMapping.size());
        auto newLoop = loopsMapping[origLoop];
        if (newLoop != ErasedLoop) {
          auto expr = mapResults[i];
          tempExprs.emplace_back(replaceAffineDim(expr, origLoop, newLoop));
        }
      }
      return mlir::AffineMap::get(finalNumLoops, 0, tempExprs, ctx);
    };

    maps.back() = updateMap(resMap);

    mlir::OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(user);

    auto loc = op.getLoc();
    llvm::SmallVector<mlir::OpFoldResult, 4> tempOffsets;
    llvm::SmallVector<mlir::OpFoldResult, 4> tempSizes;
    llvm::SmallVector<mlir::OpFoldResult, 4> tempStrides;
    for (auto i : llvm::seq<size_t>(0, oldInputs.size())) {
      assert(i < maps.size());
      auto input = oldInputs[i];
      auto inputType = input.getType().cast<mlir::RankedTensorType>();
      auto inputRank = static_cast<unsigned>(inputType.getRank());
      auto inputMap = maps[i];

      bool needView = false;
      tempOffsets.resize(inputRank);
      tempSizes.resize(inputRank);
      tempStrides.resize(inputRank);

      unsigned inputResultRank = 0;
      for (auto r : llvm::seq(0u, inputRank)) {
        auto inputDim = inputMap.getDimPosition(r);
        if (auto indVal = findResDim(inputDim)) {
          auto ind = *indVal;
          tempOffsets[r] = offsets[ind];
          tempSizes[r] = sizes[ind];
          tempStrides[r] = strides[ind];
          needView = true;
          assert(ind < droppedDims.size());
          if (!droppedDims[ind])
            ++inputResultRank;
        } else {
          tempOffsets[r] = zero;
          tempSizes[r] =
              rewriter.createOrFold<mlir::tensor::DimOp>(loc, input, r);
          tempStrides[r] = one;
          ++inputResultRank;
        }
      }

      if (needView) {
        mlir::RankedTensorType viewType;
        if (inputResultRank < inputRank) {
          viewType =
              mlir::tensor::ExtractSliceOp::inferCanonicalRankReducedResultType(
                  inputResultRank, inputType, tempOffsets, tempSizes,
                  tempStrides);
        } else {
          viewType = mlir::tensor::ExtractSliceOp::inferResultType(
              inputType, tempOffsets, tempSizes, tempStrides);
        }
        newInputs[i] = rewriter.createOrFold<mlir::tensor::ExtractSliceOp>(
            loc, viewType, input, tempOffsets, tempSizes, tempStrides);
      } else {
        newInputs[i] = input;
      }

      maps[i] = updateMap(inputMap);
    }

    auto outputType = output.getType().cast<mlir::RankedTensorType>();
    mlir::RankedTensorType newInitType;
    if (droppedDims.any()) {
      auto initRank = droppedDims.size() - droppedDims.count();
      newInitType =
          mlir::tensor::ExtractSliceOp::inferCanonicalRankReducedResultType(
              initRank, outputType, offsets, sizes, strides);
    } else {
      newInitType = mlir::tensor::ExtractSliceOp::inferResultType(
          outputType, offsets, sizes, strides);
    }

    mlir::Value newInit = rewriter.create<mlir::tensor::ExtractSliceOp>(
        loc, newInitType, output, offsets, sizes, strides);

    auto newOp = rewriter.create<mlir::linalg::GenericOp>(
        loc, newInit.getType(), newInputs, newInit, maps, newIters);
    auto &newRegion = newOp.getRegion();

    rewriter.inlineRegionBefore(op.getRegion(), newRegion, newRegion.end());

    assert(droppedDims.size() == offsets.size());
    auto updateLinagIndexOps =
        [&](mlir::Operation *innerOp) -> mlir::WalkResult {
      if (mlir::isa<mlir::linalg::GenericOp>(innerOp))
        return mlir::WalkResult::skip();

      auto indexOp = mlir::dyn_cast<mlir::linalg::IndexOp>(innerOp);
      if (!indexOp)
        return mlir::WalkResult::advance();

      uint64_t dim = indexOp.getDim();
      if (dim >= droppedDims.size())
        return mlir::WalkResult::interrupt();

      if (droppedDims[dim]) {
        mlir::OpBuilder::InsertionGuard g(rewriter);
        rewriter.setInsertionPoint(indexOp);
        auto loc = indexOp.getLoc();
        auto val =
            mlir::getValueOrCreateConstantIndexOp(rewriter, loc, offsets[dim]);
        rewriter.replaceOp(indexOp, val);
        return mlir::WalkResult::advance();
      }

      uint64_t newIndex = 0;
      for (auto i : llvm::seq<uint64_t>(0, dim)) {
        (void)i;
        if (!droppedDims[dim])
          ++newIndex;
      }
      rewriter.modifyOpInPlace(indexOp, [&]() { indexOp.setDim(newIndex); });

      return mlir::WalkResult::advance();
    };

    if (newOp.getRegion().walk(updateLinagIndexOps).wasInterrupted())
      return mlir::failure();

    mlir::Value result = newOp.getResult(0);
    if (extractElem)
      result = rewriter.create<mlir::tensor::ExtractOp>(loc, result);

    rewriter.replaceOp(user, result);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct OptimizeGlobalsConstsLoad
    : public mlir::OpRewritePattern<mlir::memref::LoadOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::LoadOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // We access data outside function, but doesnt change it, lets hope it
    // is safe.
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    if (!mod)
      return mlir::failure();

    mlir::SymbolTable symbolTable(mod);

    llvm::SmallVector<uint64_t> indices(op.getIndices().size());
    for (auto &&[i, ind] : llvm::enumerate(op.getIndices())) {
      auto val = mlir::getConstantIntValue(ind);
      if (!val || *val < 0)
        return mlir::failure();

      indices[i] = static_cast<uint64_t>(*val);
    }

    auto getGlobal = op.getMemref().getDefiningOp<mlir::memref::GetGlobalOp>();
    if (!getGlobal)
      return mlir::failure();

    auto sym = symbolTable.lookup<mlir::memref::GlobalOp>(getGlobal.getName());
    if (!sym)
      return mlir::failure();

    if (!sym.getConstant())
      return mlir::failure();

    auto initAttr = sym.getInitialValue();
    if (!initAttr)
      return mlir::failure();

    auto elements = initAttr->dyn_cast<mlir::ElementsAttr>();
    if (!elements)
      return mlir::failure();

    auto elementsType = mlir::dyn_cast<mlir::ShapedType>(elements.getType());
    if (!elementsType || elementsType.getElementType() != op.getType() ||
        !elements.isValidIndex(indices))
      return mlir::failure();

    auto vals = elements.tryGetValues<mlir::Attribute>();
    if (!vals)
      return mlir::failure();

    mlir::Attribute val = (*vals)[indices];

    if (auto complexType = mlir::dyn_cast<mlir::ComplexType>(
            op.getMemRefType().getElementType())) {
      rewriter.replaceOpWithNewOp<mlir::complex::ConstantOp>(
          op, complexType, val.cast<mlir::ArrayAttr>());
    } else {
      rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(
          op, mlir::cast<mlir::TypedAttr>(val));
    }

    return mlir::success();
  }
};

struct OptimizeSingleElemCopy
    : public mlir::OpRewritePattern<mlir::memref::CopyOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::CopyOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto src = op.getSource();
    auto srcType = src.getType().dyn_cast<mlir::MemRefType>();
    if (!srcType)
      return mlir::failure();

    auto dst = op.getTarget();
    auto dstType = src.getType().dyn_cast<mlir::MemRefType>();
    if (!dstType)
      return mlir::failure();

    if (srcType.getRank() != 1 ||
        (srcType.getShape()[0] != 1 && dstType.getShape()[0] != 1))
      return mlir::failure();

    auto loc = op->getLoc();
    mlir::Value idx = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value val = rewriter.create<mlir::memref::LoadOp>(loc, src, idx);
    rewriter.replaceOpWithNewOp<mlir::memref::StoreOp>(op, val, dst, idx);
    return mlir::success();
  }
};

struct OptimizeIdentityLayoutStrides
    : public mlir::OpRewritePattern<mlir::memref::ExtractStridedMetadataOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::ExtractStridedMetadataOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto memref = op.getSource();
    mlir::MemRefType memrefType = memref.getType();
    if (!memrefType.getLayout().isIdentity())
      return mlir::failure();

    if (llvm::all_of(op.getStrides(),
                     [](auto stride) { return stride.use_empty(); }))
      return mlir::failure();

    auto loc = op.getLoc();
    auto sizesVals = getSizes(rewriter, loc, memref);
    auto newStrides =
        computeIdentityStrides(rewriter, loc, memrefType.getShape(), sizesVals);

    for (auto &&[oldStride, newStride] :
         llvm::zip(op.getStrides(), newStrides)) {
      mlir::Value newStrideVal =
          mlir::getValueOrCreateConstantIndexOp(rewriter, loc, newStride);
      rewriter.replaceAllUsesWith(oldStride, newStrideVal);
    }

    return mlir::success();
  }
};

struct PostPlierToLinalgInnerPass
    : public mlir::PassWrapper<PostPlierToLinalgInnerPass,
                               mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PostPlierToLinalgInnerPass)

  void runOnOperation() override;
};

void PostPlierToLinalgInnerPass::runOnOperation() {
  auto &context = getContext();
  mlir::RewritePatternSet patterns(&context);

  numba::populateCommonOptsPatterns(patterns);

  patterns.insert<SimplifyExpandDims>(&context);

  if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                      std::move(patterns))))
    return signalPassFailure();
}

template <typename F>
static void visitTypeRecursive(mlir::Type type, F &&visitor) {
  if (auto tupleType = type.dyn_cast<mlir::TupleType>()) {
    for (auto t : tupleType.getTypes())
      visitTypeRecursive(t, std::forward<F>(visitor));
  } else {
    visitor(type);
  }
}

struct LinalgOptInnerPass
    : public mlir::PassWrapper<LinalgOptInnerPass, mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgOptInnerPass)

  void runOnOperation() override;
};

static const constexpr llvm::StringLiteral
    kMixedGenericNoalias("mixed_generic_noalias");

static bool defaultControlFusionFn(mlir::OpOperand *fusedOperand) {
  assert(fusedOperand);
  if (llvm::hasNItemsOrMore(fusedOperand->get().getUses(), 2))
    return false;

  if (auto generic =
          mlir::dyn_cast<mlir::linalg::GenericOp>(fusedOperand->getOwner())) {
    // Mixed generics fusion
    if (!generic.hasPureTensorSemantics() &&
        !generic.hasPureBufferSemantics()) {
      auto numInputs = generic.getNumDpsInputs();
      auto noalias = generic->getAttrOfType<mlir::DenseBoolArrayAttr>(
          kMixedGenericNoalias);
      if (!noalias || noalias.size() != numInputs)
        return false;

      auto idx = fusedOperand->getOperandNumber();

      // Do not fuse outputs.
      if (idx >= numInputs)
        return false;

      return noalias[idx];
    }
  }
  return true;
}

void LinalgOptInnerPass::runOnOperation() {
  auto &context = getContext();
  mlir::RewritePatternSet patterns(&context);

  numba::populateCommonOptsPatterns(patterns);

  patterns.insert<
      // clang-format off
      SimplifyExpandDims,
      LowerEnforceShape,
      GenerateToFill,
      // InsertSliceToPad,
      SliceOfGeneric
      // clang-format on
      >(&context);

  mlir::linalg::populateElementwiseOpsFusionPatterns(patterns,
                                                     defaultControlFusionFn);
  mlir::linalg::populateEraseUnusedOperandsAndResultsPatterns(patterns);

  if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                      std::move(patterns))))
    return signalPassFailure();
}

struct FuseAdjacentGenerics
    : public mlir::OpRewritePattern<mlir::linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::linalg::GenericOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics())
      return mlir::failure();

    llvm::SmallDenseMap<mlir::Value, mlir::Attribute> origArgs;
    auto block = op->getBlock();
    mlir::DominanceInfo dom;
    for (auto opArg : op->getOperands()) {
      for (auto user : opArg.getUsers()) {
        if (user == op)
          continue;

        if (block != user->getBlock() || !dom.properlyDominates(op, user))
          continue;

        auto other = mlir::dyn_cast<mlir::linalg::GenericOp>(user);
        if (!other || other.getIteratorTypes() != op.getIteratorTypes() ||
            !other.hasPureTensorSemantics())
          continue;

        auto dominates = [&]() -> bool {
          for (auto arg : other.getOperands())
            if (!dom.properlyDominates(arg, op))
              return false;

          return true;
        }();
        if (!dominates)
          continue;

        if (origArgs.empty()) {
          for (auto &&[arg, map] :
               llvm::zip(op.getOperands(), op.getIndexingMaps()))
            origArgs.insert({arg, map});
        }

        auto hasSameArg = [&]() -> bool {
          for (auto &&[arg, map] :
               llvm::zip(other.getOperands(), other.getIndexingMaps())) {
            auto it = origArgs.find(arg);
            if (it != origArgs.end() && it->second == map)
              return true;
          }
          return false;
        }();

        if (!hasSameArg)
          continue;

        auto concat = [](auto &&range1, auto &&range2) {
          auto ret = llvm::to_vector(range1);
          ret.append(range2.begin(), range2.end());
          return ret;
        };

        auto numArgs1 = op.getInputs().size();
        auto numArgs2 = other.getInputs().size();
        auto numRes1 = op.getOutputs().size();
        auto numRes2 = other.getOutputs().size();

        auto newInputs = concat(op.getInputs(), other.getInputs());
        auto newOutputs = concat(op.getOutputs(), other.getOutputs());

        auto getMaps = [](auto c) {
          return llvm::map_range(c, [](auto a) {
            return mlir::cast<mlir::AffineMapAttr>(a).getValue();
          });
        };

        llvm::SmallVector<mlir::AffineMap> newMaps;
        llvm::append_range(
            newMaps,
            getMaps(op.getIndexingMaps().getValue().take_front(numArgs1)));
        llvm::append_range(
            newMaps,
            getMaps(other.getIndexingMaps().getValue().take_front(numArgs2)));
        llvm::append_range(
            newMaps,
            getMaps(op.getIndexingMaps().getValue().drop_front(numArgs1)));
        llvm::append_range(
            newMaps,
            getMaps(other.getIndexingMaps().getValue().drop_front(numArgs2)));

        auto newResTypes = concat(op.getResultTypes(), other.getResultTypes());

        auto iterators = op.getIteratorTypesArray();

        auto loc = op.getLoc();

        auto bodyBuilder = [](mlir::OpBuilder &, mlir::Location,
                              mlir::ValueRange) {};
        auto newGeneric = rewriter.create<mlir::linalg::GenericOp>(
            loc, newResTypes, newInputs, newOutputs, newMaps, iterators,
            bodyBuilder);

        auto newBody = newGeneric.getBody();

        auto body1 = op.getBody();
        auto body2 = other.getBody();
        auto term1 = mlir::cast<mlir::linalg::YieldOp>(body1->getTerminator());
        auto term2 = mlir::cast<mlir::linalg::YieldOp>(body2->getTerminator());

        auto newInArgs =
            newBody->getArguments().take_front(numArgs1 + numArgs2);
        auto newOutArgs =
            newBody->getArguments().drop_front(numArgs1 + numArgs2);

        (void)numRes2;
        assert(newOutArgs.size() == (numRes1 + numRes2));

        llvm::SmallVector<mlir::Value> blockArgs;
        llvm::append_range(blockArgs, newInArgs.take_front(numArgs1));
        llvm::append_range(blockArgs, newOutArgs.take_front(numRes1));

        assert(blockArgs.size() == body1->getNumArguments());
        rewriter.inlineBlockBefore(body1, newBody, newBody->end(), blockArgs);

        blockArgs.clear();
        llvm::append_range(blockArgs, newInArgs.drop_front(numArgs1));
        llvm::append_range(blockArgs, newOutArgs.drop_front(numRes1));

        assert(blockArgs.size() == body2->getNumArguments());
        rewriter.inlineBlockBefore(body2, newBody, newBody->end(), blockArgs);

        auto newYieldArgs = concat(term1.getValues(), term2.getValues());

        rewriter.eraseOp(term1);
        rewriter.eraseOp(term2);

        mlir::OpBuilder::InsertionGuard g(rewriter);
        rewriter.setInsertionPointToEnd(newBody);
        rewriter.create<mlir::linalg::YieldOp>(loc, newYieldArgs);

        auto newResults = newGeneric.getResults();
        rewriter.replaceOp(op, newResults.take_front(numRes1));
        rewriter.replaceOp(other, newResults.drop_front(numRes1));

        return mlir::success();
      }
    }
    return mlir::failure();
  }
};

static bool checkToTensorPrevOp(mlir::Operation &op) {
  auto it = mlir::Block::iterator(op);
  auto block = op.getBlock();
  auto begin = block->begin();
  if (it == begin)
    return false;

  auto &prevOp = *std::prev(it);
  return mlir::isa<numba::ntensor::ToTensorOp>(prevOp) ||
         prevOp.hasTrait<mlir::OpTrait::ConstantLike>();
}

struct MoveToTensor
    : public mlir::OpRewritePattern<numba::ntensor::ToTensorOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(numba::ntensor::ToTensorOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (checkToTensorPrevOp(*op))
      return mlir::failure();

    auto src = op.getArray();
    if (auto defOp = src.getDefiningOp()) {
      auto it1 = mlir::Block::iterator(defOp);
      auto it2 = mlir::Block::iterator(op);
      if (std::next(it1) == it2)
        return mlir::failure();

      rewriter.modifyOpInPlace(op, [&]() { op->moveAfter(defOp); });
      return mlir::success();
    }

    auto block = mlir::cast<mlir::BlockArgument>(src).getParentBlock();
    auto begin = block->begin();
    auto it = mlir::Block::iterator(op);
    if (it == begin)
      return mlir::failure();

    auto *prevOp = &(*begin);

    rewriter.modifyOpInPlace(op, [&]() { op->moveBefore(prevOp); });
    return mlir::success();
  }
};

struct FuseAdjacentGenericsPass
    : public mlir::PassWrapper<FuseAdjacentGenericsPass,
                               mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseAdjacentGenericsPass)

  void runOnOperation() override {
    auto *context = &getContext();
    mlir::RewritePatternSet patterns(context);

    patterns.insert<FuseAdjacentGenerics, MoveToTensor>(context);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                        std::move(patterns))))
      return signalPassFailure();
  }
};

template <typename F>
static bool mayAliasImpl(mlir::Value src, F &&mayAliasCheck) {
  llvm::SmallVector<mlir::Value> worklist;
  worklist.emplace_back(src);
  do {
    auto current = worklist.pop_back_val();
    if (mayAliasCheck(current))
      return true;

    if (auto extract = current.getDefiningOp<mlir::tensor::ExtractSliceOp>())
      worklist.emplace_back(extract.getSource());

    if (auto generic = current.getDefiningOp<mlir::linalg::GenericOp>()) {
      for (auto arg : generic->getOperands()) {
        if (mlir::isa<mlir::MemRefType>(arg.getType()) && mayAliasCheck(arg))
          return true;

        worklist.emplace_back(arg);
      }
    }

    if (auto toTensor =
            current.getDefiningOp<mlir::bufferization::ToTensorOp>())
      worklist.emplace_back(toTensor.getMemref());

    if (auto enforceShape =
            current.getDefiningOp<numba::util::EnforceShapeOp>())
      worklist.emplace_back(enforceShape.getValue());

  } while (!worklist.empty());
  return false;
}

static bool mayAlias(mlir::AliasAnalysis &analysis, mlir::Value input,
                     mlir::MutableOperandRange outputs) {
  for (auto &&outOperand : outputs) {
    auto out = outOperand.get();
    auto check = [&](mlir::Value val) {
      return !analysis.alias(val, out).isNo();
    };
    if (mlir::isa<mlir::MemRefType>(out.getType()) &&
        mayAliasImpl(input, check))
      return true;
  }
  return false;
}

struct MixedGenericsAliasAnalysis
    : public mlir::PassWrapper<MixedGenericsAliasAnalysis,
                               mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MixedGenericsAliasAnalysis)

  void runOnOperation() override {
    auto &aliasAnalysis = getAnalysis<mlir::AliasAnalysis>();
    mlir::OpBuilder builder(&getContext());
    auto attrName = builder.getStringAttr(kMixedGenericNoalias);
    llvm::SmallVector<bool> flagsArray;
    auto visitor = [&](mlir::linalg::GenericOp op) {
      if (op.hasPureBufferSemantics() || op.hasPureTensorSemantics())
        return;

      flagsArray.resize(op.getNumDpsInputs());
      for (auto &&[i, arg] : llvm::enumerate(op.getDpsInputOperands()))
        flagsArray[i] =
            !mayAlias(aliasAnalysis, arg->get(), op.getDpsInitsMutable());

      auto flags = builder.getDenseBoolArrayAttr(flagsArray);
      op->setAttr(attrName, flags);
    };
    getOperation()->walk(visitor);
  }
};

struct BufferizeReshape
    : public mlir::OpConversionPattern<numba::util::ReshapeOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(numba::util::ReshapeOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto getType = [&](mlir::Type type) {
      auto shapedType = mlir::cast<mlir::ShapedType>(type);
      return mlir::MemRefType::get(shapedType.getShape(),
                                   shapedType.getElementType());
    };
    auto source = adaptor.getSource();
    auto shape = adaptor.getShape();
    auto resType = getType(op.getType());
    rewriter.replaceOpWithNewOp<numba::util::ReshapeOp>(op, resType, source,
                                                        shape);
    return mlir::success();
  }
};

struct BufferizeExtractSlice
    : public mlir::OpConversionPattern<mlir::tensor::ExtractSliceOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::tensor::ExtractSliceOp op,
                  mlir::tensor::ExtractSliceOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    assert(converter);
    auto dstType = converter->convertType(op.getType())
                       .dyn_cast_or_null<mlir::MemRefType>();
    if (!dstType)
      return mlir::failure();

    auto src = adaptor.getSource();
    auto srcType = src.getType().cast<mlir::MemRefType>();

    auto dstRank = dstType.getRank();
    auto offsets = mlir::getMixedValues(adaptor.getStaticOffsets(),
                                        adaptor.getOffsets(), rewriter);
    auto sizes = mlir::getMixedValues(adaptor.getStaticSizes(),
                                      adaptor.getSizes(), rewriter);
    auto strides = mlir::getMixedValues(adaptor.getStaticStrides(),
                                        adaptor.getStrides(), rewriter);

    auto viewType = [&]() {
      if (srcType.getRank() == dstRank)
        return mlir::memref::SubViewOp::inferResultType(srcType, offsets, sizes,
                                                        strides)
            .cast<mlir::MemRefType>();

      return mlir::memref::SubViewOp::inferRankReducedResultType(
                 dstType.getShape(), srcType, offsets, sizes, strides)
          .cast<mlir::MemRefType>();
    }();
    auto loc = op->getLoc();
    mlir::Value view = rewriter.create<mlir::memref::SubViewOp>(
        loc, viewType, src, offsets, sizes, strides);

    if (viewType != dstType)
      view = rewriter.create<numba::util::ChangeLayoutOp>(loc, dstType, view);

    rewriter.replaceOp(op, view);
    return mlir::success();
  }
};

static mlir::Value genCopy(mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::Value src) {
  auto srcType = mlir::cast<mlir::MemRefType>(src.getType());
  llvm::SmallVector<mlir::Value> sizes;
  for (auto &&[i, s] : llvm::enumerate(srcType.getShape())) {
    if (!mlir::ShapedType::isDynamic(s))
      continue;

    mlir::Value dim = builder.create<mlir::memref::DimOp>(loc, src, i);
    sizes.emplace_back(dim);
  }

  auto resType = mlir::MemRefType::get(
      srcType.getShape(), srcType.getElementType(),
      mlir::MemRefLayoutAttrInterface{}, srcType.getMemorySpace());
  mlir::Value res = builder.create<mlir::memref::AllocOp>(loc, resType, sizes);
  builder.create<mlir::memref::CopyOp>(loc, src, res);
  return res;
}

struct BufferizeMixedGeneric
    : public mlir::OpConversionPattern<mlir::linalg::GenericOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::linalg::GenericOp op,
                  mlir::linalg::GenericOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (op.hasPureTensorSemantics() || op.hasPureBufferSemantics())
      return mlir::failure();

    auto noalias =
        op->getAttrOfType<mlir::DenseBoolArrayAttr>(kMixedGenericNoalias);
    if (!noalias || noalias.size() != op.getNumDpsInputs())
      return mlir::failure();

    auto loc = op.getLoc();
    bool changed = false;
    mlir::ValueRange inputs = adaptor.getInputs();
    llvm::SmallVector<mlir::Value> newInputs(inputs.size());
    for (auto &&[i, input] : llvm::enumerate(inputs)) {
      auto orig = op.getInputs()[i];
      if (mlir::isa<mlir::RankedTensorType>(orig.getType())) {
        changed = true;
        mlir::Value arg = input;
        if (!noalias[i])
          arg = genCopy(rewriter, loc, arg);

        newInputs[i] = arg;
      } else {
        newInputs[i] = input;
      }
    }

    mlir::ValueRange outputs = adaptor.getOutputs();
    llvm::SmallVector<mlir::Value> newResults;
    for (auto &&[i, output] : llvm::enumerate(outputs)) {
      auto orig = op.getOutputs()[i];
      if (mlir::isa<mlir::RankedTensorType>(orig.getType())) {
        changed = true;
        newResults.emplace_back(output);
      }
    }

    if (!changed)
      return mlir::failure();

    auto newOp = rewriter.create<mlir::linalg::GenericOp>(
        loc, std::nullopt, newInputs, outputs, adaptor.getIndexingMaps(),
        adaptor.getIteratorTypes(), nullptr, nullptr);

    auto &newRegion = newOp.getRegion();

    auto &srcRegion = op.getRegion();
    rewriter.inlineRegionBefore(srcRegion, newRegion, newRegion.end());
    rewriter.replaceOp(op, newResults);
    return mlir::success();
  }
};

struct BufferizeSignCast
    : public mlir::OpConversionPattern<numba::util::SignCastOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(numba::util::SignCastOp op,
                  numba::util::SignCastOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    assert(getTypeConverter() && "Invalid type converter");
    auto resType = getTypeConverter()->convertType(op.getType());
    if (!resType)
      return mlir::failure();

    rewriter.replaceOpWithNewOp<numba::util::SignCastOp>(op, resType,
                                                         adaptor.getSource());
    return mlir::success();
  }
};

struct AdditionalBufferize
    : public mlir::PassWrapper<AdditionalBufferize, mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AdditionalBufferize)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<numba::util::NumbaUtilDialect>();
  }

  void runOnOperation() override {
    auto *context = &getContext();

    mlir::bufferization::BufferizeTypeConverter typeConverter;
    numba::populateTupleTypeConverter(typeConverter);

    auto materializeTupleCast =
        [](mlir::OpBuilder &builder, mlir::Type type, mlir::ValueRange inputs,
           mlir::Location loc) -> std::optional<mlir::Value> {
      if (inputs.size() != 1)
        return std::nullopt;

      auto input = inputs.front();
      if (input.getType().isa<mlir::TupleType>() || type.isa<mlir::TupleType>())
        return builder
            .create<mlir::UnrealizedConversionCastOp>(loc, type, input)
            .getResult(0);

      return std::nullopt;
    };
    typeConverter.addArgumentMaterialization(materializeTupleCast);
    typeConverter.addSourceMaterialization(materializeTupleCast);
    typeConverter.addTargetMaterialization(materializeTupleCast);

    mlir::RewritePatternSet patterns(context);
    mlir::ConversionTarget target(*context);

    numba::populateControlFlowTypeConversionRewritesAndTarget(typeConverter,
                                                              patterns, target);
    numba::populateTupleTypeConversionRewritesAndTarget(typeConverter, patterns,
                                                        target);
    target
        .addIllegalOp<mlir::tensor::ReshapeOp, mlir::tensor::ExtractSliceOp>();
    target.addLegalOp<mlir::memref::ReshapeOp>();
    target
        .addDynamicallyLegalOp<numba::util::SignCastOp, numba::util::ReshapeOp>(
            [&](mlir::Operation *op) { return typeConverter.isLegal(op); });

    target.addDynamicallyLegalOp<mlir::linalg::GenericOp>(
        [](mlir::linalg::GenericOp op) {
          return op.hasPureTensorSemantics() || op.hasPureBufferSemantics();
        });

    patterns.insert<BufferizeReshape, BufferizeExtractSlice,
                    BufferizeMixedGeneric, BufferizeSignCast>(typeConverter,
                                                              context);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

struct GenAtomicAdd : public mlir::OpRewritePattern<mlir::memref::StoreOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::StoreOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto atomicOp = op.getValueToStore().getDefiningOp();
    if (!atomicOp)
      return mlir::failure();

    auto reg = isInsideAtomicRegion(op);
    if (!reg)
      return mlir::failure();

    auto memref = op.getMemRef();
    if (!mlir::DominanceInfo().properlyDominates(memref, reg))
      return mlir::failure();

    auto indices = op.getIndices();

    auto checkAddOp = [&](auto addOp) -> mlir::Value {
      for (bool reverse : {false, true}) {
        auto load = (reverse ? addOp.getLhs() : addOp.getRhs())
                        .template getDefiningOp<mlir::memref::LoadOp>();
        if (!load)
          continue;

        if (load.getMemRef() != memref || load.getIndices() != indices)
          continue;

        return (reverse ? addOp.getRhs() : addOp.getLhs());
      }
      return nullptr;
    };

    if (auto addOp = mlir::dyn_cast<mlir::arith::AddIOp>(atomicOp)) {
      auto other = checkAddOp(addOp);
      if (!other)
        return mlir::failure();

      rewriter.create<mlir::memref::AtomicRMWOp>(
          op.getLoc(), mlir::arith::AtomicRMWKind::addi, other, memref,
          indices);
      rewriter.eraseOp(op);
      return mlir::success();
    }
    if (auto addOp = mlir::dyn_cast<mlir::arith::AddFOp>(atomicOp)) {
      auto other = checkAddOp(addOp);
      if (!other)
        return mlir::failure();

      rewriter.create<mlir::memref::AtomicRMWOp>(
          op.getLoc(), mlir::arith::AtomicRMWKind::addf, other, memref,
          indices);
      rewriter.eraseOp(op);
      return mlir::success();
    }
    if (auto addOp = mlir::dyn_cast<mlir::complex::AddOp>(atomicOp)) {
      auto other = checkAddOp(addOp);
      if (!other)
        return mlir::failure();

      auto loc = op.getLoc();

      auto srcType = mlir::cast<mlir::MemRefType>(memref.getType());
      auto elemType = mlir::cast<mlir::ComplexType>(srcType.getElementType())
                          .getElementType();
      auto rank = srcType.getShape().size();
      llvm::SmallVector<mlir::OpFoldResult> offsets(rank);
      llvm::copy(indices, offsets.begin());

      llvm::SmallVector<mlir::OpFoldResult> sizes(rank,
                                                  rewriter.getIndexAttr(1));

      mlir::Value view = rewriter.create<mlir::memref::SubViewOp>(
          loc, memref, offsets, sizes, sizes);
      auto resType =
          mlir::MemRefType::get(2, elemType, mlir::MemRefLayoutAttrInterface{},
                                srcType.getMemorySpace());
      view =
          rewriter.create<numba::util::MemrefApplyOffsetOp>(loc, resType, view);

      auto re = rewriter.create<mlir::complex::ReOp>(loc, other);
      auto im = rewriter.create<mlir::complex::ImOp>(loc, other);

      mlir::Value zero = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
      mlir::Value one = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
      rewriter.create<mlir::memref::AtomicRMWOp>(
          loc, mlir::arith::AtomicRMWKind::addf, re, view, zero);
      rewriter.create<mlir::memref::AtomicRMWOp>(
          loc, mlir::arith::AtomicRMWKind::addf, im, view, one);

      rewriter.eraseOp(op);
      return mlir::success();
    }

    return mlir::failure();
  }
};

struct GenAtomicOpsPass
    : public numba::RewriteWrapperPass<GenAtomicOpsPass, void, void,
                                       GenAtomicAdd> {};

struct RemoveAtomicRegions
    : public mlir::OpRewritePattern<numba::util::EnvironmentRegionOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(numba::util::EnvironmentRegionOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!mlir::isa<numba::util::AtomicAttr>(op.getEnvironment()))
      return mlir::failure();

    auto visitor = [](mlir::memref::StoreOp) -> mlir::WalkResult {
      return mlir::WalkResult::interrupt();
    };
    if (op.getRegion().walk(visitor).wasInterrupted())
      return mlir::failure();

    numba::util::EnvironmentRegionOp::inlineIntoParent(rewriter, op);
    return mlir::success();
  }
};

struct RemoveParallelRegions
    : public mlir::OpRewritePattern<numba::util::EnvironmentRegionOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(numba::util::EnvironmentRegionOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!mlir::isa<numba::util::ParallelAttr>(op.getEnvironment()))
      return mlir::failure();

    auto visitor = [](mlir::Operation *op) -> mlir::WalkResult {
      if (mlir::isa<mlir::scf::WhileOp, mlir::scf::ForOp,
                    mlir::scf::ParallelOp>(op))
        return mlir::WalkResult::interrupt();

      return mlir::WalkResult::advance();
    };
    if (op.getRegion().walk(visitor).wasInterrupted())
      return mlir::failure();

    numba::util::EnvironmentRegionOp::inlineIntoParent(rewriter, op);
    return mlir::success();
  }
};

// TODO: move to gpu dialect caonicalizations
struct RemoveGPURegions
    : public mlir::OpRewritePattern<numba::util::EnvironmentRegionOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(numba::util::EnvironmentRegionOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!mlir::isa<gpu_runtime::GPURegionDescAttr>(op.getEnvironment()))
      return mlir::failure();

    auto visitor = [](mlir::Operation *op) -> mlir::WalkResult {
      if (mlir::isa<mlir::memref::SubViewOp, mlir::memref::DimOp>(op))
        return mlir::WalkResult::advance();

      if (mlir::isa<mlir::scf::WhileOp, mlir::scf::ForOp,
                    mlir::scf::ParallelOp>(op))
        return mlir::WalkResult::interrupt();

      if (mlir::isa<mlir::gpu::GPUDialect, mlir::memref::MemRefDialect,
                    gpu_runtime::GpuRuntimeDialect>(op->getDialect()))
        return mlir::WalkResult::interrupt();

      return mlir::WalkResult::advance();
    };
    if (op.getRegion().walk(visitor).wasInterrupted())
      return mlir::failure();

    numba::util::EnvironmentRegionOp::inlineIntoParent(rewriter, op);
    return mlir::success();
  }
};

struct CleanupRegionsPass
    : public numba::RewriteWrapperPass<
          CleanupRegionsPass, void, void, RemoveAtomicRegions,
          RemoveParallelRegions, RemoveGPURegions> {};

struct RemoveAllAtomicRegions
    : public mlir::OpRewritePattern<numba::util::EnvironmentRegionOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(numba::util::EnvironmentRegionOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!mlir::isa<numba::util::AtomicAttr>(op.getEnvironment()))
      return mlir::failure();

    numba::util::EnvironmentRegionOp::inlineIntoParent(rewriter, op);
    return mlir::success();
  }
};

struct RemoveAtomicRegionsPass
    : public numba::RewriteWrapperPass<GenAtomicOpsPass, void, void,
                                       RemoveAllAtomicRegions> {};

struct MarkArgsRestrictPass
    : public mlir::PassWrapper<MarkArgsRestrictPass,
                               mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MarkArgsRestrictPass)

  void runOnOperation() override {
    mlir::OpBuilder builder(&getContext());
    auto attrName = builder.getStringAttr(numba::getRestrictArgName());
    auto attr = builder.getUnitAttr();
    getOperation()->walk([&](mlir::FunctionOpInterface func) {
      auto numArgs = func.getNumArguments();
      for (auto i : llvm::seq(0u, numArgs)) {
        func.setArgAttr(i, attrName, attr);
      }
    });
  }
};

struct EnforceUniqueResultsOwnershipPass
    : public mlir::PassWrapper<EnforceUniqueResultsOwnershipPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      EnforceUniqueResultsOwnershipPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::bufferization::BufferizationDialect>();
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<mlir::scf::SCFDialect>();
  }

  void runOnOperation() override {
    auto op = getOperation();
    if (op.isPrivate() || op.isDeclaration() || op.isExternal() ||
        op.getNumResults() <= 1)
      return markAllAnalysesPreserved();

    bool changed = false;
    mlir::OpBuilder builder(&getContext());
    for (mlir::Block &block : op.getFunctionBody()) {
      auto ret = mlir::dyn_cast<mlir::func::ReturnOp>(block.getTerminator());
      if (!ret)
        continue;

      mlir::Location loc = ret.getLoc();
      builder.setInsertionPoint(ret);
      mlir::ValueRange args = ret.getOperands();
      auto newArgs = llvm::to_vector(args);
      bool updated = false;
      for (auto &&[i, arg] : llvm::enumerate(args)) {
        auto type = mlir::dyn_cast<mlir::MemRefType>(arg.getType());
        if (!type)
          continue;

        mlir::Value token;
        mlir::Value needClone;
        for (auto &&other : args.drop_front(i + 1)) {
          if (!mlir::isa<mlir::MemRefType>(other.getType()))
            continue;

          if (!token)
            token =
                builder.create<mlir::memref::ExtractAlignedPointerAsIndexOp>(
                    loc, arg);

          mlir::Value otherToken =
              builder.create<mlir::memref::ExtractAlignedPointerAsIndexOp>(
                  loc, other);
          mlir::Value cmp = builder.create<mlir::arith::CmpIOp>(
              loc, mlir::arith::CmpIPredicate::eq, token, otherToken);
          if (!needClone) {
            needClone = cmp;
          } else {
            needClone = builder.create<mlir::arith::OrIOp>(loc, needClone, cmp);
          }
        }

        if (!needClone)
          continue;

        changed = true;
        updated = true;

        // TODO: to workaround stupid C++ spec limitation on capturing
        // structured bindings into lambda.
        mlir::Value argCopy = arg;
        auto trueBuilder = [&](mlir::OpBuilder &builder, mlir::Location loc) {
          mlir::Value cloned =
              builder.create<mlir::bufferization::CloneOp>(loc, argCopy);
          builder.create<mlir::scf::YieldOp>(loc, cloned);
        };
        auto falseBuilder = [&](mlir::OpBuilder &builder, mlir::Location loc) {
          builder.create<mlir::scf::YieldOp>(loc, argCopy);
        };
        auto ifOp = builder.create<mlir::scf::IfOp>(loc, needClone, trueBuilder,
                                                    falseBuilder);
        newArgs[i] = ifOp.getResult(0);
      }

      if (!updated)
        continue;

      ret.getOperandsMutable().assign(newArgs);
    }

    if (!changed)
      return markAllAnalysesPreserved();
  }
};

struct GenerateAllocTokens
    : public mlir::PassWrapper<GenerateAllocTokens, mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenerateAllocTokens)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<numba::util::NumbaUtilDialect>();
  }

  void runOnOperation() override {
    mlir::OpBuilder builder(&getContext());

    bool changed = false;
    auto visitor = [&](mlir::memref::ExtractAlignedPointerAsIndexOp op) {
      changed = true;
      builder.setInsertionPoint(op);
      mlir::Value res = builder.create<numba::util::GetAllocTokenOp>(
          op.getLoc(), op.getSource());
      op->replaceAllUsesWith(mlir::ValueRange(res));
      op->erase();
    };
    getOperation()->walk(visitor);

    if (!changed)
      return markAllAnalysesPreserved();
  }
};

struct ReplaceClones
    : public mlir::OpRewritePattern<mlir::bufferization::CloneOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::bufferization::CloneOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Value src = op.getSource();
    mlir::Type srcType = src.getType();
    mlir::Type dstType = op.getType();

    auto loc = op.getLoc();
    if (srcType != dstType)
      src = rewriter.create<numba::util::ChangeLayoutOp>(loc, dstType, src);

    rewriter.replaceOpWithNewOp<numba::util::RetainOp>(op, dstType, src);
    return mlir::success();
  }
};

struct LowerCloneOpsPass
    : public numba::RewriteWrapperPass<LowerCloneOpsPass, void, void,
                                       ReplaceClones> {};

struct ReplaceMemrefCopy : public mlir::OpRewritePattern<mlir::memref::CopyOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::CopyOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::linalg::makeMemRefCopyOp(rewriter, op.getLoc(), op.getSource(),
                                   op.getTarget());
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct LowerCopyOpsPass
    : public numba::RewriteWrapperPass<LowerCopyOpsPass, void, void,
                                       ReplaceMemrefCopy> {};

struct PostLinalgOptInnerPass
    : public mlir::PassWrapper<PostLinalgOptInnerPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PostLinalgOptInnerPass)

  void runOnOperation() override;
};

void PostLinalgOptInnerPass::runOnOperation() {
  auto func = getOperation();
  auto optLevel = getOptLevel(func);
  if (0 == optLevel)
    return;

  auto &context = getContext();
  mlir::RewritePatternSet patterns(&context);

  numba::populateCommonOptsPatterns(patterns);

  patterns.insert<OptimizeGlobalsConstsLoad, OptimizeSingleElemCopy,
                  OptimizeIdentityLayoutStrides>(&context);

  auto additionalOpt = [](mlir::func::FuncOp op) {
    auto check = [](mlir::Operation &op) -> bool {
      return mlir::isa<mlir::scf::ParallelOp, numba::util::EnvironmentRegionOp>(
          op);
    };
    (void)numba::prepareForFusion(op.getRegion(), check);
    return numba::naivelyFuseParallelOps(op.getRegion());
  };

  if (mlir::failed(applyOptimizations(func, std::move(patterns),
                                      getAnalysisManager(), additionalOpt)))
    signalPassFailure();
}

struct MoveTrivialIntoRegionPass
    : public mlir::PassWrapper<MoveTrivialIntoRegionPass,
                               mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MoveTrivialIntoRegionPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::math::MathDialect>();
    registry.insert<numba::util::NumbaUtilDialect>();
  }

  void runOnOperation() override {
    auto root = getOperation();

    auto checkOp = [](mlir::Operation *op) -> bool {
      if (op->hasTrait<mlir::OpTrait::IsTerminator>() || !mlir::isPure(op) ||
          !op->getRegions().empty())
        return false;

      if (op->hasTrait<mlir::OpTrait::ConstantLike>())
        return false;

      if (mlir::isa<mlir::ViewLikeOpInterface>(op))
        return true;

      auto dialect = op->getDialect();
      if (mlir::isa<mlir::arith::ArithDialect, mlir::math::MathDialect,
                    numba::util::NumbaUtilDialect>(dialect))
        return true;

      return false;
    };

    llvm::SmallVector<mlir::Operation *> opsToCheck;
    root->walk([&](mlir::Operation *op) {
      if (checkOp(op))
        opsToCheck.emplace_back(op);
    });

    for (auto *op : llvm::reverse(opsToCheck)) {
      assert(op);

      auto block = op->getBlock();
      auto iter = op->getIterator();
      if (iter == block->end())
        continue;

      auto region = mlir::dyn_cast<numba::util::EnvironmentRegionOp>(*(++iter));
      if (!region)
        continue;

      bool isUsedOutside = false;
      for (auto user : op->getUsers()) {
        if (!region->isProperAncestor(user)) {
          isUsedOutside = true;
          break;
        }
      }

      if (isUsedOutside)
        continue;

      mlir::Operation &firstOp = region.getRegion().front().front();
      op->moveBefore(&firstOp);
    }
  }
};

/// Move reduction iterators to the right to help later reduction simplification
/// passes.
struct MakeGenericReduceInnermost
    : public mlir::OpRewritePattern<mlir::linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::linalg::GenericOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto iters = op.getIteratorTypesArray();

    auto numDims = static_cast<unsigned>(iters.size());
    llvm::SmallBitVector reductions(iters.size());
    bool seenReduction = false;
    bool needChange = false;
    for (auto &&[i, iter] : llvm::enumerate(iters)) {
      if (iter == mlir::utils::IteratorType::reduction) {
        reductions[i] = true;
        seenReduction = true;
      } else {
        if (seenReduction)
          needChange = true;
      }
    }

    if (!needChange)
      return mlir::failure();

    llvm::SmallVector<mlir::Attribute> remappedIters;
    remappedIters.reserve(numDims);

    llvm::SmallVector<mlir::AffineExpr> remappedDims(numDims);

    auto ctx = getContext();
    unsigned newIdx = 0;
    for (bool redLoop : {false, true}) {
      for (auto i : llvm::seq(0u, numDims)) {
        if (redLoop == reductions[i]) {
          remappedIters.emplace_back(
              mlir::linalg::IteratorTypeAttr::get(ctx, iters[i]));
          remappedDims[i] = mlir::getAffineDimExpr(newIdx, ctx);
          newIdx++;
        }
      }
    }

    auto maps = op.getIndexingMaps();
    llvm::SmallVector<mlir::Attribute> newMaps;
    newMaps.reserve(maps.size());

    for (auto attr : maps.getAsRange<mlir::AffineMapAttr>()) {
      auto map = attr.getAffineMap();
      auto newMap =
          map.replaceDimsAndSymbols(remappedDims, std::nullopt, numDims, 0);
      newMaps.emplace_back(mlir::AffineMapAttr::get(newMap));
    }

    rewriter.modifyOpInPlace(op, [&]() {
      op.setIndexingMapsAttr(rewriter.getArrayAttr(newMaps));
      op.setIteratorTypesAttr(rewriter.getArrayAttr(remappedIters));
    });

    return mlir::success();
  }
};

struct MakeGenericReduceInnermostPass
    : public numba::RewriteWrapperPass<MakeGenericReduceInnermostPass, void,
                                       void, MakeGenericReduceInnermost> {};

/// Later passes (e.g. buffer deallocation) may not know how to handle poison
/// memrefs. Replace them with dummy zero-size allocations.
struct ReplaceMemrefPoison : public mlir::OpRewritePattern<mlir::ub::PoisonOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::ub::PoisonOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto resType = mlir::dyn_cast<mlir::MemRefType>(op.getType());
    if (!resType)
      return rewriter.notifyMatchFailure(op, "Not a memref");

    auto loc = op.getLoc();
    llvm::SmallVector<int64_t> shape(resType.getRank(), 0);
    auto dummyType = mlir::MemRefType::get(shape, resType.getElementType(),
                                           mlir::MemRefLayoutAttrInterface{},
                                           resType.getMemorySpace());
    mlir::Value memref = rewriter.create<mlir::memref::AllocOp>(loc, dummyType);
    if (resType != dummyType)
      memref = rewriter.create<mlir::memref::CastOp>(loc, resType, memref);

    rewriter.replaceOp(op, memref);
    return mlir::success();
  }
};

struct ReplaceMemrefPoisonPass
    : public numba::RewriteWrapperPass<ReplaceMemrefPoisonPass, void, void,
                                       ReplaceMemrefPoison> {};

struct InsertParallelRegionPass
    : public mlir::PassWrapper<InsertParallelRegionPass,
                               mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InsertParallelRegionPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<numba::util::NumbaUtilDialect>();
  }

  void runOnOperation() override {
    llvm::SmallSetVector<mlir::scf::WhileOp, 8> loops;
    getOperation()->walk([&](plier::IternextOp op) {
      auto loop = mlir::dyn_cast<mlir::scf::WhileOp>(op->getParentOp());
      if (!loop)
        return;

      auto getiter = op.getValue().getDefiningOp<plier::GetiterOp>();
      if (!getiter)
        return;

      auto call = getiter.getValue().getDefiningOp<plier::PyCallOp>();
      if (!call)
        return;

      auto name = call.getFuncName();
      if (name != "numba.prange")
        return;

      loops.insert(loop);
    });

    if (loops.empty())
      return markAllAnalysesPreserved();

    auto *ctx = &getContext();
    auto env = numba::util::ParallelAttr::get(ctx);
    mlir::OpBuilder builder(ctx);
    for (auto loop : loops) {
      auto loc = loop.getLoc();
      builder.setInsertionPoint(loop);
      auto region = builder.create<numba::util::EnvironmentRegionOp>(
          loc, env, /*args*/ std::nullopt, loop->getResultTypes());
      mlir::Block &body = region.getRegion().front();
      body.getTerminator()->erase();
      loop.getResults().replaceAllUsesWith(region.getResults());
      builder.setInsertionPointToEnd(&body);
      auto term = builder.create<numba::util::EnvironmentRegionYieldOp>(
          loc, loop.getResults());
      loop->moveBefore(term);
    }
  }
};

struct GenAtomicRegion : public mlir::OpRewritePattern<plier::InplaceBinOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(plier::InplaceBinOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!isInsideParallelRegion(op) || isInsideAtomicRegion(op))
      return mlir::failure();

    auto loc = op.getLoc();
    auto env = numba::util::AtomicAttr::get(getContext());
    auto reg = rewriter.create<numba::util::EnvironmentRegionOp>(
        loc, env, std::nullopt, op.getType());
    auto body = reg.getBody();
    rewriter.eraseOp(body->getTerminator());

    mlir::OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(body);

    auto res = op.getResult();
    auto newTerm =
        rewriter.create<numba::util::EnvironmentRegionYieldOp>(loc, res);
    rewriter.modifyOpInPlace(op, [&]() { op->moveBefore(newTerm); });
    for (auto user : op->getUsers()) {
      if (!mlir::isa<plier::SetItemOp>(user))
        continue;

      rewriter.modifyOpInPlace(user, [&]() { user->moveBefore(newTerm); });
    }

    auto checker = [&](mlir::OpOperand &arg) {
      return arg.getOwner()->getParentOp() != reg;
    };
    rewriter.replaceUsesWithIf(res, reg.getResult(0), checker);
    return mlir::success();
  }
};

struct GenAtomicRegionPass
    : public numba::RewriteWrapperPass<GenAtomicRegionPass, void, void,
                                       GenAtomicRegion> {};

static void populateCommonOptPass(mlir::OpPassManager &pm) {
  pm.addPass(numba::createCompositePass(
      "PlierToLinalgCommonOptPass", [](mlir::OpPassManager &p) {
        p.addNestedPass<mlir::func::FuncOp>(
            std::make_unique<MoveTrivialIntoRegionPass>());
        p.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
        p.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
      }));
}

static void populateDeallocationPipeline(mlir::OpPassManager &pm) {
  mlir::bufferization::BufferDeallocationPipelineOptions deallocOpts;

  // TODO: breaks private declarations
  deallocOpts.privateFunctionDynamicOwnership = false;
  mlir::bufferization::buildBufferDeallocationPipeline(pm, deallocOpts);
  pm.addNestedPass<mlir::func::FuncOp>(
      std::make_unique<EnforceUniqueResultsOwnershipPass>());
  pm.addNestedPass<mlir::func::FuncOp>(std::make_unique<LowerCloneOpsPass>());
  pm.addNestedPass<mlir::func::FuncOp>(std::make_unique<GenerateAllocTokens>());
  pm.addPass(numba::createCompositePass(
      "PostDeallocCleanups", [](mlir::OpPassManager &p) {
        p.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
        p.addNestedPass<mlir::func::FuncOp>(numba::createCommonOptsPass());
      }));
}

static void populatePlierToLinalgRegionPipeline(mlir::OpPassManager &pm) {
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(std::make_unique<InsertParallelRegionPass>());
  pm.addPass(std::make_unique<GenAtomicRegionPass>());
}

static void populatePlierToLinalgGenPipeline(mlir::OpPassManager &pm) {
  pm.addPass(std::make_unique<MarkArgsRestrictPass>());
  pm.addPass(std::make_unique<PlierToNtensorPass>());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  pm.addPass(std::make_unique<ResolveNumpyFuncsPass>());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::func::FuncOp>(numba::createCopyRemovalPass());
  populateCommonOptPass(pm);
  pm.addPass(numba::createPromoteWhilePass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(numba::ntensor::createPropagateEnvironmentPass());
  pm.addPass(std::make_unique<ResolveNtensorPass>());
  pm.addPass(numba::createForceInlinePass());
  pm.addPass(mlir::createSymbolDCEPass());
  populateCommonOptPass(pm);
  pm.addNestedPass<mlir::func::FuncOp>(
      std::make_unique<WrapParforRegionsPass>());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::func::FuncOp>(numba::createNtensorAliasAnalysisPass());
  pm.addNestedPass<mlir::func::FuncOp>(numba::createNtensorToLinalgPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      numba::createNtensorLowerToTensorCopyPass());
  pm.addNestedPass<mlir::func::FuncOp>(numba::createCopyRemovalPass());
  pm.addPass(std::make_unique<MarkInputShapesRanges>());
  pm.addPass(numba::createCompositePass(
      "PostPlierToLinalgPass", [](mlir::OpPassManager &p) {
        p.addPass(numba::createShapeIntegerRangePropagationPass());
        p.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
        p.addNestedPass<mlir::func::FuncOp>(
            std::make_unique<PostPlierToLinalgInnerPass>());
      }));
}

static void populatePlierToLinalgOptPipeline(mlir::OpPassManager &pm) {
  pm.addPass(
      numba::createCompositePass("LinalgOptPass", [](mlir::OpPassManager &p) {
        p.addPass(numba::createShapeIntegerRangePropagationPass());
        p.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
        p.addNestedPass<mlir::func::FuncOp>(
            std::make_unique<MixedGenericsAliasAnalysis>());
        p.addNestedPass<mlir::func::FuncOp>(
            std::make_unique<LinalgOptInnerPass>());
        p.addNestedPass<mlir::func::FuncOp>(
            std::make_unique<FuseAdjacentGenericsPass>());
        p.addPass(numba::createRemoveUnusedArgsPass());
      }));

  pm.addPass(numba::createNtensorToMemrefPass());
  pm.addPass(numba::createExpandTuplePass());
  pm.addPass(mlir::createCanonicalizerPass());

  pm.addPass(numba::createMakeSignlessPass());
  pm.addPass(mlir::createCanonicalizerPass());

  pm.addPass(mlir::createReconcileUnrealizedCastsPass());

  pm.addPass(mlir::arith::createConstantBufferizePass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::bufferization::createEmptyTensorToAllocTensorPass());
  pm.addPass(mlir::createCanonicalizerPass());

  pm.addNestedPass<mlir::func::FuncOp>(
      std::make_unique<MixedGenericsAliasAnalysis>());
  pm.addNestedPass<mlir::func::FuncOp>(std::make_unique<AdditionalBufferize>());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createSCFBufferizePass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createLinalgBufferizePass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::tensor::createTensorBufferizePass());
  pm.addPass(mlir::func::createFuncBufferizePass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::bufferization::createFinalizingBufferizePass());

  pm.addPass(mlir::createCanonicalizerPass());

  //  pm.addNestedPass<mlir::func::FuncOp>(
  //      mlir::bufferization::createBufferHoistingPass());
  //  pm.addNestedPass<mlir::func::FuncOp>(
  //      mlir::bufferization::createBufferLoopHoistingPass());

  pm.addPass(std::make_unique<MakeStridedLayoutPass>());
  pm.addPass(std::make_unique<OptimizeStridedLayoutPass>());
  pm.addNestedPass<mlir::func::FuncOp>(
      std::make_unique<FinalizeStridedLayoutPass>());

  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());

  //  pm.addNestedPass<mlir::func::FuncOp>(
  //      mlir::bufferization::createPromoteBuffersToStackPass());

  pm.addNestedPass<mlir::func::FuncOp>(
      std::make_unique<MakeGenericReduceInnermostPass>());
  pm.addNestedPass<mlir::func::FuncOp>(std::make_unique<LowerCopyOpsPass>());
  pm.addNestedPass<mlir::func::FuncOp>(numba::createCopyRemovalPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::createConvertLinalgToParallelLoopsPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      std::make_unique<ReplaceMemrefPoisonPass>());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());

  pm.addPass(numba::createForceInlinePass());
  pm.addPass(mlir::createSymbolDCEPass());

  pm.addPass(numba::createPromoteBoolMemrefPass());
  pm.addNestedPass<mlir::func::FuncOp>(numba::createUpliftMathPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::createLoopInvariantCodeMotionPass());

  pm.addPass(numba::createShapeIntegerRangePropagationPass());
  pm.addPass(std::make_unique<MarkArgsRestrictPass>());
  pm.addNestedPass<mlir::func::FuncOp>(
      std::make_unique<PropagateFastmathFlags>());
  pm.addPass(numba::createCompositePass(
      "PostLinalgOptPass", [](mlir::OpPassManager &p) {
        p.addPass(numba::createNormalizeMemrefArgsPass());
        p.addNestedPass<mlir::func::FuncOp>(
            std::make_unique<GenAtomicOpsPass>());
        p.addNestedPass<mlir::func::FuncOp>(
            std::make_unique<CleanupRegionsPass>());
        p.addNestedPass<mlir::func::FuncOp>(
            std::make_unique<MoveTrivialIntoRegionPass>());
        p.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
        p.addNestedPass<mlir::func::FuncOp>(
            numba::createCanonicalizeReductionsPass());
        p.addNestedPass<mlir::func::FuncOp>(
            numba::createPromoteToParallelPass());
        p.addNestedPass<mlir::func::FuncOp>(
            numba::createMoveIntoParallelPass());
        // ToDo: This pass also tries to do some simple fusion, whic should be
        // split in separate pass
        p.addNestedPass<mlir::func::FuncOp>(
            std::make_unique<PostLinalgOptInnerPass>());
        p.addPass(numba::createRemoveUnusedArgsPass());
      }));

  pm.addNestedPass<mlir::func::FuncOp>(
      std::make_unique<RemoveAtomicRegionsPass>());
  pm.addNestedPass<mlir::func::FuncOp>(
      std::make_unique<PropagateFastmathFlags>());
  // Uplifting FMAs can interfere with other optimizations, like loop reduction
  // uplifting. Move it after main optimization pass.
  pm.addNestedPass<mlir::func::FuncOp>(mlir::math::createMathUpliftToFMA());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  populateDeallocationPipeline(pm);

  pm.addPass(mlir::createSymbolDCEPass());
}
} // namespace

// ToDo: how does this sink stuff actually works?
void registerPlierToLinalgPipeline(numba::PipelineRegistry &registry) {
  registry.registerPipeline([](auto sink) {
    auto stage = getHighLoweringStage();
    sink(plierToLinalgRegionPipelineName(),
         {stage.begin, plierToScfPipelineName()},
         {stage.end, plierToLinalgGenPipelineName(), plierToStdPipelineName(),
          untuplePipelineName()},
         {}, &populatePlierToLinalgRegionPipeline);

    sink(plierToLinalgGenPipelineName(), {plierToStdPipelineName()},
         {plierToLinalgOptPipelineName(), untuplePipelineName()},
         {plierToScfPipelineName()}, &populatePlierToLinalgGenPipeline);

    sink(plierToLinalgOptPipelineName(),
         {plierToLinalgGenPipelineName(), untuplePipelineName()},
         {removeSignPipelineName(), stage.end}, {},
         &populatePlierToLinalgOptPipeline);
  });
}

llvm::StringRef plierToLinalgRegionPipelineName() {
  return "plier_to_linalg_region";
}

llvm::StringRef plierToLinalgGenPipelineName() { return "plier_to_linalg_gen"; }

llvm::StringRef plierToLinalgOptPipelineName() { return "plier_to_linalg_opt"; }
