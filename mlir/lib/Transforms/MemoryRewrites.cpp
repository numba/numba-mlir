// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "numba/Transforms/MemoryRewrites.hpp"

#include "numba/Analysis/MemorySsaAnalysis.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dominance.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace {
struct Meminfo {
  mlir::Value memref;
  mlir::ValueRange indices;

  bool operator==(const Meminfo &other) const {
    return memref == other.memref && indices == other.indices;
  }
};

static std::optional<Meminfo> getMeminfo(mlir::Operation *op) {
  assert(nullptr != op);
  if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(op))
    return Meminfo{load.getMemref(), load.getIndices()};

  if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(op))
    return Meminfo{store.getMemref(), store.getIndices()};

  if (auto load = mlir::dyn_cast<mlir::vector::LoadOp>(op))
    return Meminfo{load.getBase(), load.getIndices()};

  if (auto store = mlir::dyn_cast<mlir::vector::StoreOp>(op))
    return Meminfo{store.getBase(), store.getIndices()};

  return {};
}

static mlir::Value getStoreValue(mlir::Operation *op) {
  assert(op);
  if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(op))
    return store.getValue();

  if (auto store = mlir::dyn_cast<mlir::vector::StoreOp>(op))
    return store.getValueToStore();

  llvm_unreachable("Invalid store op");
}

struct MustAlias {
  bool operator()(mlir::Operation *op1, mlir::Operation *op2) const {
    auto meminfo1 = getMeminfo(op1);
    if (!meminfo1)
      return false;

    auto meminfo2 = getMeminfo(op2);
    if (!meminfo2)
      return false;

    return *meminfo1 == *meminfo2;
  }
};

static mlir::LogicalResult
optimizeUses(numba::MemorySSAAnalysis &memSSAAnalysis) {
  return memSSAAnalysis.optimizeUses();
}

static mlir::LogicalResult foldLoads(numba::MemorySSAAnalysis &memSSAAnalysis) {
  assert(memSSAAnalysis.memssa);
  auto &memSSA = *memSSAAnalysis.memssa;
  using NodeType = numba::MemorySSA::NodeType;
  bool changed = false;

  mlir::DominanceInfo dom;
  for (auto &node : llvm::make_early_inc_range(memSSA.getNodes())) {
    if (NodeType::Use == memSSA.getNodeType(&node)) {
      auto op1 = memSSA.getNodeOperation(&node);
      assert(nullptr != op1);
      if (op1->getNumResults() != 1)
        continue;

      auto def = memSSA.getNodeDef(&node);
      assert(nullptr != def);
      if (NodeType::Def != memSSA.getNodeType(def))
        continue;

      auto op2 = memSSA.getNodeOperation(def);
      assert(nullptr != op2);
      if (!MustAlias()(op1, op2))
        continue;

      auto val = getStoreValue(op2);
      auto res = op1->getResult(0);
      if (val.getType() != res.getType())
        continue;

      if (!dom.properlyDominates(val, op1))
        continue;

      res.replaceAllUsesWith(val);
      op1->erase();
      memSSA.eraseNode(&node);
      changed = true;
    }
  }
  return mlir::success(changed);
}

static mlir::LogicalResult
deadStoreElemination(numba::MemorySSAAnalysis &memSSAAnalysis) {
  assert(memSSAAnalysis.memssa);
  auto &memSSA = *memSSAAnalysis.memssa;
  using NodeType = numba::MemorySSA::NodeType;
  auto getNextDef =
      [&](numba::MemorySSA::Node *node) -> numba::MemorySSA::Node * {
    numba::MemorySSA::Node *def = nullptr;
    for (auto user : memSSA.getUsers(node)) {
      auto type = memSSA.getNodeType(user);
      if (NodeType::Def == type) {
        if (def != nullptr)
          return nullptr;

        def = user;
      } else {
        return nullptr;
      }
    }
    return def;
  };
  bool changed = false;
  for (auto &node : llvm::make_early_inc_range(memSSA.getNodes())) {
    if (NodeType::Def == memSSA.getNodeType(&node)) {
      if (auto nextDef = getNextDef(&node)) {
        auto op1 = memSSA.getNodeOperation(&node);
        auto op2 = memSSA.getNodeOperation(nextDef);
        assert(nullptr != op1);
        assert(nullptr != op2);
        if (MustAlias()(op1, op2)) {
          op1->erase();
          memSSA.eraseNode(&node);
          changed = true;
        }
      }
    }
  }
  return mlir::success(changed);
}

struct SimpleOperationInfo : public llvm::DenseMapInfo<mlir::Operation *> {
  static unsigned getHashValue(const mlir::Operation *opC) {
    return static_cast<unsigned>(mlir::OperationEquivalence::computeHash(
        const_cast<mlir::Operation *>(opC),
        mlir::OperationEquivalence::directHashValue,
        mlir::OperationEquivalence::ignoreHashValue,
        mlir::OperationEquivalence::IgnoreLocations));
  }
  static bool isEqual(const mlir::Operation *lhsC,
                      const mlir::Operation *rhsC) {
    auto *lhs = const_cast<mlir::Operation *>(lhsC);
    auto *rhs = const_cast<mlir::Operation *>(rhsC);
    if (lhs == rhs)
      return true;
    if (lhs == getTombstoneKey() || lhs == getEmptyKey() ||
        rhs == getTombstoneKey() || rhs == getEmptyKey())
      return false;
    return mlir::OperationEquivalence::isEquivalentTo(
        lhs, rhs, mlir::OperationEquivalence::exactValueMatch, nullptr,
        mlir::OperationEquivalence::IgnoreLocations);
  }
};

static mlir::LogicalResult loadCSE(numba::MemorySSAAnalysis &memSSAAnalysis) {
  mlir::DominanceInfo dom;
  assert(memSSAAnalysis.memssa);
  auto &memSSA = *memSSAAnalysis.memssa;
  using NodeType = numba::MemorySSA::NodeType;
  bool changed = false;
  llvm::SmallDenseMap<mlir::Operation *, mlir::Operation *, 4,
                      SimpleOperationInfo>
      opsMap;
  for (auto &node : memSSA.getNodes()) {
    auto nodeType = memSSA.getNodeType(&node);
    if (NodeType::Def != nodeType && NodeType::Phi != nodeType &&
        NodeType::Root != nodeType)
      continue;

    opsMap.clear();
    for (auto user : memSSA.getUsers(&node)) {
      if (memSSA.getNodeType(user) != NodeType::Use)
        continue;

      auto op = memSSA.getNodeOperation(user);
      if (!op->getRegions().empty())
        continue;

      auto it = opsMap.find(op);
      if (it == opsMap.end()) {
        opsMap.insert({op, op});
      } else {
        auto firstUser = it->second;
        if (!MustAlias()(op, firstUser))
          continue;

        if (dom.properlyDominates(op, firstUser)) {
          firstUser->replaceAllUsesWith(op);
          opsMap[firstUser] = op;
          auto firstUserNode = memSSA.getNode(firstUser);
          assert(firstUserNode);
          memSSA.eraseNode(firstUserNode);
          firstUser->erase();
          changed = true;
        } else if (dom.properlyDominates(firstUser, op)) {
          op->replaceAllUsesWith(firstUser);
          op->erase();
          memSSA.eraseNode(user);
          changed = true;
        }
      }
    }
  }
  return mlir::success(changed);
}

} // namespace

std::optional<mlir::LogicalResult>
numba::optimizeMemoryOps(mlir::AnalysisManager &am) {
  auto &memSSAAnalysis = am.getAnalysis<MemorySSAAnalysis>();
  if (!memSSAAnalysis.memssa)
    return {};

  using fptr_t = mlir::LogicalResult (*)(MemorySSAAnalysis &);
  const fptr_t funcs[] = {
      &optimizeUses,
      &foldLoads,
      &deadStoreElemination,
      &loadCSE,
  };

  bool changed = false;
  bool repeat = false;

  do {
    repeat = false;
    for (auto func : funcs) {
      if (mlir::succeeded(func(memSSAAnalysis))) {
        changed = true;
        repeat = true;
      }
    }
  } while (repeat);

  return mlir::success(changed);
}

namespace {
struct RemoveDeadAllocs
    : public mlir::OpInterfaceRewritePattern<mlir::MemoryEffectOpInterface> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::MemoryEffectOpInterface op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!op.onlyHasEffect<mlir::MemoryEffects::Allocate>() ||
        op->getNumResults() != 1)
      return mlir::failure();

    auto res = op->getResult(0);
    for (auto user : op->getUsers()) {
      if (user->getNumResults() != 0)
        return mlir::failure();

      auto memInterface = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(user);
      if (!memInterface)
        return mlir::failure();

      if (!memInterface.getEffectOnValue<mlir::MemoryEffects::Free>(res) &&
          !memInterface.getEffectOnValue<mlir::MemoryEffects::Write>(res))
        return mlir::failure();

      if (memInterface.getEffectOnValue<mlir::MemoryEffects::Read>(res))
        return mlir::failure();
    }

    for (auto user : llvm::make_early_inc_range(op->getUsers()))
      rewriter.eraseOp(user);

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct MemoryOptPass
    : public mlir::PassWrapper<MemoryOptPass,
                               mlir::InterfacePass<mlir::FunctionOpInterface>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MemoryOptPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<mlir::vector::VectorDialect>();
  }

  void runOnOperation() override {
    auto *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);

    patterns.insert<RemoveDeadAllocs>(ctx);

    mlir::FrozenRewritePatternSet fPatterns(std::move(patterns));
    auto am = getAnalysisManager();
    while (true) {
      if (mlir::failed(
              mlir::applyPatternsAndFoldGreedily(getOperation(), fPatterns)))
        return signalPassFailure();

      am.invalidate({});
      auto res = numba::optimizeMemoryOps(am);
      if (!res) {
        getOperation()->emitError("Failed to build memory SSA analysis");
        return signalPassFailure();
      }
      if (mlir::failed(*res))
        break;
    }
  }
};

static int64_t mergeDims(int64_t dim1, int64_t dim2, bool isDst) {
  if (dim1 == dim2)
    return dim1;

  if (isDst) {
    if (!mlir::ShapedType::isDynamic(dim2))
      return dim2;

    if (!mlir::ShapedType::isDynamic(dim1))
      return dim1;
  }

  return mlir::ShapedType::kDynamic;
}

static std::optional<mlir::MemRefLayoutAttrInterface>
mergeLayouts(mlir::MemRefLayoutAttrInterface layout1,
             mlir::MemRefLayoutAttrInterface layout2, bool isDst) {
  if (layout1 == layout2)
    return layout1;

  auto strided1 = mlir::dyn_cast<mlir::StridedLayoutAttr>(layout1);
  auto strided2 = mlir::dyn_cast<mlir::StridedLayoutAttr>(layout1);
  if (!strided1 || !strided2)
    return std::nullopt;

  assert(strided1.getStrides().size() == strided2.getStrides().size());
  auto offset = mergeDims(strided1.getOffset(), strided2.getOffset(), isDst);

  llvm::SmallVector<int64_t> strides;
  for (auto &&[stride1, stride2] :
       llvm::zip(strided1.getStrides(), strided2.getStrides()))
    strides.emplace_back(mergeDims(stride1, stride2, isDst));

  return mlir::StridedLayoutAttr::get(layout1.getContext(), offset, strides);
}

static std::optional<mlir::MemRefType>
mergeMemrefTypes(mlir::MemRefType type1, mlir::MemRefType type2, bool isDst) {
  if (type1.getElementType() != type2.getElementType() ||
      type1.getMemorySpace() != type2.getMemorySpace())
    return std::nullopt;

  auto layout = mergeLayouts(type1.getLayout(), type2.getLayout(), isDst);
  if (!layout)
    return std::nullopt;

  assert(type1.getRank() == type2.getRank());
  llvm::SmallVector<int64_t> shape;
  shape.reserve(type1.getRank());
  for (auto &&[dim1, dim2] : llvm::zip(type1.getShape(), type2.getShape()))
    shape.emplace_back(mergeDims(dim1, dim2, isDst));

  return mlir::MemRefType::get(shape, type1.getElementType(), *layout,
                               type1.getMemorySpace());
}

struct NormalizeMemrefArgs
    : public mlir::PassWrapper<NormalizeMemrefArgs,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NormalizeMemrefArgs)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::memref::MemRefDialect>();
  }

  void runOnOperation() override {
    auto mod = getOperation();

    bool changed = false;

    mlir::OpBuilder builder(&getContext());
    llvm::SmallVector<mlir::CallOpInterface> calls;
    llvm::SmallVector<mlir::Type> newTypes;
    mlir::IRMapping mapping;
    for (auto func : mod.getOps<mlir::FunctionOpInterface>()) {
      if (func.isDeclaration() || func.isPublic())
        continue;

      if (!mlir::isa<mlir::FunctionType>(func.getFunctionType()))
        continue;

      auto args = func.getArgumentTypes();

      bool hasMemrefArgs = false;
      for (auto arg : args)
        hasMemrefArgs = hasMemrefArgs || mlir::isa<mlir::MemRefType>(arg);

      if (!hasMemrefArgs)
        continue;

      auto funcUses = mlir::SymbolTable::getSymbolUses(func, mod);
      if (!funcUses)
        continue;

      calls.clear();
      bool unknownUser = false;
      for (auto use : *funcUses) {
        auto user = use.getUser();
        auto callOp = mlir::dyn_cast<mlir::CallOpInterface>(user);
        if (!callOp) {
          unknownUser = true;
          break;
        }

        calls.emplace_back(callOp);
      }

      if (unknownUser)
        continue;

      bool typesChanged = false;
      newTypes.clear();
      for (auto &&[i, arg] : llvm::enumerate(args)) {
        if (!mlir::isa<mlir::MemRefType>(arg)) {
          newTypes.emplace_back(arg);
          continue;
        }

        mlir::MemRefType newType;
        for (auto call : calls) {
          auto arg = call.getArgOperands()[i];
          if (auto cast = arg.getDefiningOp<mlir::memref::CastOp>())
            arg = cast.getSource();

          auto type = mlir::cast<mlir::MemRefType>(arg.getType());
          if (!newType) {
            newType = type;
            continue;
          }

          auto result = mergeMemrefTypes(newType, type, false);
          if (!result) {
            newType = nullptr;
            break;
          }
          newType = *result;
        }
        if (newType) {
          if (auto result = mergeMemrefTypes(
                  newType, mlir::cast<mlir::MemRefType>(arg), true))
            newType = *result;
        }

        if (newType && newType != arg) {
          typesChanged = true;
          newTypes.emplace_back(newType);
        } else {
          newTypes.emplace_back(arg);
        }
      }

      if (!typesChanged)
        continue;

      changed = true;
      for (auto call : calls) {
        mapping.clear();
        builder.setInsertionPoint(call);
        auto loc = call.getLoc();
        for (auto &&[srcType, dstType, srcArg] :
             llvm::zip(args, newTypes, call.getArgOperands())) {
          assert(dstType);
          if (dstType == srcType)
            continue;

          auto arg =
              builder.createOrFold<mlir::memref::CastOp>(loc, dstType, srcArg);
          mapping.map(srcArg, arg);
        }
        auto newOp = builder.clone(*call, mapping);
        call->replaceAllUsesWith(newOp->getResults());
        call->erase();
      }

      auto funcType = func.getFunctionType().cast<mlir::FunctionType>();
      auto newFuncType = funcType.clone(newTypes, funcType.getResults());
      func.setFunctionTypeAttr(mlir::TypeAttr::get(newFuncType));
      auto &entryBlock = func.getFunctionBody().front();
      assert(entryBlock.getNumArguments() == newTypes.size());
      builder.setInsertionPointToStart(&entryBlock);
      auto loc = builder.getUnknownLoc();
      for (auto &&[srcArg, dstType] :
           llvm::zip(entryBlock.getArguments(), newTypes)) {
        auto srcType = srcArg.getType();
        if (srcType == dstType)
          continue;

        auto cast = builder.create<mlir::memref::CastOp>(loc, srcType, srcArg);
        srcArg.replaceAllUsesExcept(cast.getResult(), cast);
        srcArg.setType(dstType);
      }
    }

    if (!changed)
      markAllAnalysesPreserved();
  }
};
} // namespace

std::unique_ptr<mlir::Pass> numba::createMemoryOptPass() {
  return std::make_unique<MemoryOptPass>();
}

std::unique_ptr<mlir::Pass> numba::createNormalizeMemrefArgsPass() {
  return std::make_unique<NormalizeMemrefArgs>();
}
