// SPDX-FileCopyrightText: 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "numba/Transforms/ShapeIntegerRangePropagation.hpp"

#include "numba/Dialect/ntensor/IR/NTensorOps.hpp"
#include "numba/Dialect/numba_util/Dialect.hpp"

#include <llvm/Support/Debug.h>
#include <mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h>
#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/Analysis/DataFlow/IntegerRangeAnalysis.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Arith/Transforms/Passes.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Interfaces/ShapedOpInterfaces.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#define DEBUG_TYPE "numba-shape-range-propagation"

namespace {
static auto getIndexRange(int64_t smin, int64_t smax) {
  unsigned width = mlir::IndexType::kInternalStorageBitWidth;
  return mlir::ConstantIntRanges::fromSigned(llvm::APInt(width, smin),
                                             llvm::APInt(width, smax));
}

static auto getDefaultDimRange() {
  return getIndexRange(0, std::numeric_limits<int64_t>::max());
}

static auto getFixedDimRange(int64_t val) { return getIndexRange(val, val); }

class ShapeValue {
public:
  ShapeValue() = default;
  ShapeValue(mlir::ShapedType shaped) : shapeRanges(std::in_place) {
    shapeRanges->reserve(shaped.getRank());
    for (auto dim : shaped.getShape()) {
      shapeRanges->emplace_back(mlir::ShapedType::isDynamic(dim)
                                    ? getDefaultDimRange()
                                    : getFixedDimRange(dim));
    }
  }
  ShapeValue(mlir::ArrayAttr attr) : shapeRanges(std::in_place) {
    shapeRanges->reserve(attr.size());
    for (auto elem : attr) {
      auto range = mlir::cast<numba::util::IndexRangeAttr>(elem);
      shapeRanges->emplace_back(getIndexRange(range.getMin(), range.getMax()));
    }
  }
  ShapeValue(mlir::ArrayRef<mlir::ConstantIntRanges> values)
      : shapeRanges(std::in_place) {
    shapeRanges->assign(values.begin(), values.end());
  }

  /// Whether the state is uninitialized.
  bool isUninitialized() const { return !shapeRanges; }

  llvm::ArrayRef<mlir::ConstantIntRanges> getShape() const {
    assert(!isUninitialized());
    return *shapeRanges;
  }

  static ShapeValue join(const ShapeValue &lhs, const ShapeValue &rhs) {
    if (lhs.isUninitialized())
      return rhs;

    if (rhs.isUninitialized())
      return lhs;

    llvm::SmallVector<mlir::ConstantIntRanges> resShapes;
    resShapes.reserve(
        std::min(lhs.shapeRanges->size(), rhs.shapeRanges->size()));
    for (auto &&[l, r] : llvm::zip(*lhs.shapeRanges, *rhs.shapeRanges))
      resShapes.emplace_back(l.rangeUnion(r));

    ShapeValue ret;
    ret.shapeRanges = std::move(resShapes);
    return ret;
  }

  static ShapeValue intersect(const ShapeValue &lhs, const ShapeValue &rhs) {
    if (lhs.isUninitialized())
      return rhs;

    if (rhs.isUninitialized())
      return lhs;

    llvm::SmallVector<mlir::ConstantIntRanges> resShapes;
    resShapes.reserve(
        std::min(lhs.shapeRanges->size(), rhs.shapeRanges->size()));
    for (auto &&[l, r] : llvm::zip(*lhs.shapeRanges, *rhs.shapeRanges))
      resShapes.emplace_back(l.intersection(r));

    ShapeValue ret;
    ret.shapeRanges = std::move(resShapes);
    return ret;
  }

  bool operator==(const ShapeValue &rhs) const {
    return shapeRanges == rhs.shapeRanges;
  }

  void print(llvm::raw_ostream &os) const {
    if (isUninitialized()) {
      os << "None";
    } else {
      os << "[";
      llvm::interleaveComma(*shapeRanges, os);
      os << "]";
    }
  }

private:
  std::optional<llvm::SmallVector<mlir::ConstantIntRanges>> shapeRanges;
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const ShapeValue &state) {
  state.print(os);
  return os;
}

struct ShapeValueLattice : public mlir::dataflow::Lattice<ShapeValue> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ShapeValueLattice)
  using Lattice::Lattice;

  void onUpdate(mlir::DataFlowSolver *solver) const override {
    Lattice::onUpdate(solver);

    auto value = point.get<mlir::Value>();
    auto *cv =
        solver->getOrCreateState<Lattice<mlir::dataflow::ConstantValue>>(value);
    return solver->propagateIfChanged(
        cv, cv->join(mlir::dataflow::ConstantValue::getUnknownConstant()));
  }
};

static bool isShapedCast(mlir::Operation *op) {
  if (mlir::isa<numba::ntensor::FromTensorOp, numba::ntensor::ToTensorOp,
                numba::ntensor::ToTensorCopyOp, numba::ntensor::FromMemrefOp,
                numba::ntensor::ToMemrefOp>(op))
    return true;

  return mlir::isa<mlir::CastOpInterface>(op) && op->getNumOperands() == 1 &&
         op->getNumResults() == 1 &&
         mlir::isa<mlir::ShapedType>(op->getOperand(0).getType()) &&
         mlir::isa<mlir::ShapedType>(op->getResult(0).getType());
}

class ShapeValueAnalysis
    : public mlir::dataflow::SparseForwardDataFlowAnalysis<ShapeValueLattice> {
public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  void visitOperation(mlir::Operation *op,
                      llvm::ArrayRef<const ShapeValueLattice *> operands,
                      llvm::ArrayRef<ShapeValueLattice *> results) override {
    LLVM_DEBUG(llvm::dbgs()
               << "ShapeValueAnalysis: Visiting operation: " << *op << "\n");

    if (auto sizesInterface =
            mlir::dyn_cast<mlir::OffsetSizeAndStrideOpInterface>(op)) {
      if (op->getNumResults() != 1)
        return;

      auto result = op->getResult(0);
      auto shaped = mlir::dyn_cast<mlir::ShapedType>(result.getType());
      if (!shaped)
        return;

      auto resultShape = shaped.getShape();
      auto mixedSizes = sizesInterface.getMixedSizes();

      llvm::SmallBitVector droppedDims(mixedSizes.size());
      unsigned shapePos = 0;
      for (const auto &size : enumerate(mixedSizes)) {
        auto sizeVal = getConstantIntValue(size.value());
        // If the size is not 1, or if the current matched dimension of the
        // result is the same static shape as the size value (which is 1), then
        // the dimension is preserved.
        if (!sizeVal || *sizeVal != 1 ||
            (shapePos < resultShape.size() && resultShape[shapePos] == 1)) {
          shapePos++;
          continue;
        }
        droppedDims.set(size.index());
      }

      llvm::SmallVector<mlir::ConstantIntRanges> ranges;
      ranges.reserve(mixedSizes.size());
      for (auto &&[i, size] : llvm::enumerate(mixedSizes)) {
        if (droppedDims[i])
          continue;

        if (auto val = mlir::getConstantIntValue(size)) {
          ranges.emplace_back(getFixedDimRange(*val));
        } else {
          assert(size.is<mlir::Value>());
          auto state = getOrCreateFor<mlir::dataflow::IntegerValueRangeLattice>(
              op, size.get<mlir::Value>());

          if (!state)
            return;

          auto value = state->getValue();
          if (value.isUninitialized())
            return;

          ranges.emplace_back(value.getValue());
        }
      }

      auto newVal = ShapeValue::intersect({shaped}, {ranges});

      LLVM_DEBUG(llvm::dbgs()
                 << "ShapeValueAnalysis: New view result: " << newVal << "\n");

      auto resultLattice = results.front();
      auto changed = resultLattice->join(newVal);
      propagateIfChanged(resultLattice, changed);
      return;
    }

    if (auto reshape = mlir::dyn_cast<numba::util::ReshapeOp>(op)) {
      auto dstShaped =
          mlir::dyn_cast<mlir::ShapedType>(reshape.getResult().getType());
      if (!dstShaped)
        return;

      auto args = reshape.getShape();

      llvm::SmallVector<mlir::ConstantIntRanges> ranges;
      ranges.reserve(args.size());
      for (auto arg : args) {
        auto state =
            getOrCreateFor<mlir::dataflow::IntegerValueRangeLattice>(op, arg);

        if (!state)
          return;

        auto value = state->getValue();
        if (value.isUninitialized())
          return;

        ranges.emplace_back(value.getValue());
      }

      auto newVal = ShapeValue::intersect({ranges}, {dstShaped});

      LLVM_DEBUG(llvm::dbgs() << "ShapeValueAnalysis: New reshape result: "
                              << newVal << "\n");

      auto resultLattice = results.front();
      auto changed = resultLattice->join(newVal);
      propagateIfChanged(resultLattice, changed);
      return;
    }

    if (auto enforceShape = mlir::dyn_cast<numba::util::EnforceShapeOp>(op)) {
      auto srcShaped =
          mlir::dyn_cast<mlir::ShapedType>(enforceShape.getValue().getType());
      if (!srcShaped)
        return;

      auto dstShaped =
          mlir::dyn_cast<mlir::ShapedType>(enforceShape.getResult().getType());
      if (!dstShaped)
        return;

      auto args = enforceShape.getSizes();

      llvm::SmallVector<mlir::ConstantIntRanges> ranges;
      ranges.reserve(args.size());
      for (auto arg : args) {
        auto state =
            getOrCreateFor<mlir::dataflow::IntegerValueRangeLattice>(op, arg);

        if (!state)
          return;

        auto value = state->getValue();
        if (value.isUninitialized())
          return;

        ranges.emplace_back(value.getValue());
      }

      ShapeValue newVal(ranges);
      newVal = ShapeValue::intersect(newVal, {srcShaped});
      newVal = ShapeValue::intersect(newVal, {dstShaped});

      auto inputLatticeVal = operands.front()->getValue();
      if (inputLatticeVal.isUninitialized())
        return;

      newVal = ShapeValue::intersect(newVal, inputLatticeVal);

      LLVM_DEBUG(llvm::dbgs()
                 << "ShapeValueAnalysis: New enforce shape result: " << newVal
                 << "\n");

      auto resultLattice = results.front();
      auto changed = resultLattice->join(newVal);
      propagateIfChanged(resultLattice, changed);
      return;
    }

    if (auto empty = mlir::dyn_cast<mlir::tensor::EmptyOp>(op)) {
      assert(results.size() == 1);
      auto mixedSizes = empty.getMixedSizes();

      llvm::SmallVector<mlir::ConstantIntRanges> ranges;
      ranges.reserve(mixedSizes.size());
      for (auto size : mixedSizes) {
        if (auto val = mlir::getConstantIntValue(size)) {
          ranges.emplace_back(getFixedDimRange(*val));
        } else {
          assert(size.is<mlir::Value>());
          auto state = getOrCreateFor<mlir::dataflow::IntegerValueRangeLattice>(
              op, size.get<mlir::Value>());

          if (!state)
            return;

          auto value = state->getValue();
          if (value.isUninitialized())
            return;

          ranges.emplace_back(value.getValue());
        }
      }

      ShapeValue newVal(ranges);

      auto shaped = mlir::cast<mlir::ShapedType>(empty.getResult().getType());
      newVal = ShapeValue::intersect(newVal, {shaped});

      LLVM_DEBUG(llvm::dbgs()
                 << "ShapeValueAnalysis: New tensor empty shape result: "
                 << newVal << "\n");

      auto resultLattice = results.front();
      auto changed = resultLattice->join(newVal);
      propagateIfChanged(resultLattice, changed);
      return;
    }

    if (auto empty = mlir::dyn_cast<numba::ntensor::CreateArrayOp>(op)) {
      assert(results.size() == 1);
      auto mixedSizes = empty.getMixedSizes();

      llvm::SmallVector<mlir::ConstantIntRanges> ranges;
      ranges.reserve(mixedSizes.size());
      for (auto size : mixedSizes) {
        if (auto val = mlir::getConstantIntValue(size)) {
          ranges.emplace_back(getFixedDimRange(*val));
        } else {
          assert(size.is<mlir::Value>());
          auto state = getOrCreateFor<mlir::dataflow::IntegerValueRangeLattice>(
              op, size.get<mlir::Value>());

          if (!state)
            return;

          auto value = state->getValue();
          if (value.isUninitialized())
            return;

          ranges.emplace_back(value.getValue());
        }
      }

      ShapeValue newVal(ranges);

      auto shaped = mlir::cast<mlir::ShapedType>(empty.getResult().getType());
      newVal = ShapeValue::intersect(newVal, {shaped});

      LLVM_DEBUG(llvm::dbgs()
                 << "ShapeValueAnalysis: New CreateArrayOp shape result: "
                 << newVal << "\n");

      auto resultLattice = results.front();
      auto changed = resultLattice->join(newVal);
      propagateIfChanged(resultLattice, changed);
      return;
    }

    if (auto generic = mlir::dyn_cast<mlir::linalg::GenericOp>(op)) {
      if (generic->getNumResults() == 0)
        return;

      bool allIdentity =
          llvm::all_of(generic.getIndexingMaps(), [](mlir::Attribute map) {
            return mlir::cast<mlir::AffineMapAttr>(map).getValue().isIdentity();
          });

      bool allParallel =
          llvm::all_of(generic.getIteratorTypes(), [](mlir::Attribute map) {
            return mlir::cast<mlir::linalg::IteratorTypeAttr>(map).getValue() ==
                   mlir::utils::IteratorType::parallel;
          });

      auto inputsNum = generic.getInputs().size();
      auto outs = generic.getOutputs();

      mlir::SmallVector<ShapeValue> newVals(outs.size());

      for (size_t i = 0; i < outs.size(); ++i) {
        auto resOperandShape = operands[inputsNum + i]->getValue();

        if (!resOperandShape.isUninitialized()) {
          newVals[i] = resOperandShape;
        } else if (auto shaped =
                       mlir::dyn_cast<mlir::ShapedType>(outs[i].getType())) {
          newVals[i] = ShapeValue(shaped);
        }
      }

      if (allIdentity && allParallel) {
        // if all indexing maps are identity all outputs must be of the same
        // shape
        ShapeValue newVal = [&]() {
          for (auto &&v : newVals)
            if (!v.isUninitialized())
              return v;

          return ShapeValue();
        }();

        if (!newVal.isUninitialized()) {
          for (auto &&v : newVals)
            if (!v.isUninitialized())
              newVal = ShapeValue::intersect(newVal, v);
        }

        for (auto arg : op->getOperands())
          newVal = ShapeValue::intersect(
              newVal, {mlir::cast<mlir::ShapedType>(arg.getType())});

        for (auto result : op->getResults())
          newVal = ShapeValue::intersect(
              newVal, {mlir::cast<mlir::ShapedType>(result.getType())});

        for (auto input : operands) {
          auto inputLatticeVal = input->getValue();
          if (inputLatticeVal.isUninitialized())
            return;

          newVal = ShapeValue::intersect(newVal, inputLatticeVal);
        }

        // Since all indexing maps are identity we can propagate single result
        // to all outputs
        if (!newVal.isUninitialized()) {
          for (auto &v : newVals)
            v = ShapeValue::intersect(v, newVal);
        }
      }

      auto debug_msg = [&]() {
        llvm::dbgs() << "ShapeValueAnalysis: Shaped linalg generic: ";
        for (auto &&v : newVals)
          llvm::dbgs() << v << " ";

        llvm::dbgs() << "\n";
      };

      LLVM_DEBUG(debug_msg());

      assert(results.size() == newVals.size());
      for (size_t i = 0; i < results.size(); ++i) {
        auto changed = results[i]->join(newVals[i]);
        propagateIfChanged(results[i], changed);
      }
      return;
    }

    if (isShapedCast(op)) {
      assert(operands.size() == 1);
      assert(results.size() == 1);

      auto inputLatticeVal = operands.front()->getValue();
      if (inputLatticeVal.isUninitialized())
        return;

      auto srcShaped =
          mlir::cast<mlir::ShapedType>(op->getOperand(0).getType());
      auto dstShaped = mlir::cast<mlir::ShapedType>(op->getResult(0).getType());
      auto res =
          ShapeValue::intersect(ShapeValue{srcShaped}, ShapeValue{dstShaped});
      res = ShapeValue::intersect(inputLatticeVal, res);

      LLVM_DEBUG(llvm::dbgs()
                 << "ShapeValueAnalysis: Shaped cast: " << res << "\n");

      auto *resultLattice = results.front();
      auto changed = resultLattice->join(res);
      propagateIfChanged(resultLattice, changed);
      return;
    }

    if (auto select = mlir::dyn_cast<mlir::arith::SelectOp>(op)) {
      if (!mlir::isa<mlir::ShapedType>(select.getResult().getType()))
        return;

      assert(operands.size() == 3);
      assert(results.size() == 1);
      auto lhs = operands[1]->getValue();
      auto rhs = operands[2]->getValue();
      if (lhs.isUninitialized() || rhs.isUninitialized())
        return;

      auto newVal = ShapeValue::join(lhs, rhs);

      LLVM_DEBUG(llvm::dbgs()
                 << "ShapeValueAnalysis: select: " << newVal << "\n");

      auto resultLattice = results.front();
      auto changed = resultLattice->join(newVal);
      propagateIfChanged(resultLattice, changed);
      return;
    }

    for (auto &&[res, resultLattice] : llvm::zip(op->getResults(), results)) {
      auto shaped = mlir::dyn_cast<mlir::ShapedType>(res.getType());
      if (!shaped)
        continue;

      auto newVal = ShapeValue{shaped};
      LLVM_DEBUG(llvm::dbgs()
                 << "ShapeValueAnalysis: New result val: " << newVal << "\n");

      auto changed = resultLattice->join(newVal);
      propagateIfChanged(resultLattice, changed);
    }
  }

  void setToEntryState(ShapeValueLattice *lattice) override {
    propagateIfChanged(lattice, lattice->join(ShapeValue{}));
  }

  mlir::LogicalResult initialize(mlir::Operation *top) override {
    if (mlir::failed(SparseForwardDataFlowAnalysis::initialize(top)))
      return mlir::failure();

    auto attrName = mlir::StringAttr::get(
        top->getContext(), numba::util::attributes::getShapeRangeName());
    top->walk([&](mlir::FunctionOpInterface func) {
      if (func.isExternal())
        return;

      auto &body = func.getFunctionBody();
      assert(!body.empty());
      for (auto &&[i, arg] : llvm::enumerate(body.front().getArguments())) {
        auto shaped = mlir::dyn_cast<mlir::ShapedType>(arg.getType());
        if (!shaped)
          continue;

        auto ind = static_cast<unsigned>(i);
        auto newRange = [&]() -> std::optional<ShapeValue> {
          auto attr = func.getArgAttrOfType<mlir::ArrayAttr>(ind, attrName);
          if (attr)
            return ShapeValue::intersect({shaped}, {attr});

          auto mod = func->getParentOfType<mlir::ModuleOp>();
          if (!mod)
            return std::nullopt;

          auto uses = mlir::SymbolTable::getSymbolUses(func, mod);
          if (!uses || !uses->empty())
            return std::nullopt;

          return ShapeValue{shaped};
        }();

        if (!newRange)
          continue;

        LLVM_DEBUG(llvm::dbgs() << "ShapeValueAnalysis: initialize: " << arg
                                << " " << *newRange << "\n");

        auto *lattice = getLatticeElement(arg);
        assert(lattice);
        assert(lattice->getValue().isUninitialized());
        propagateIfChanged(lattice, lattice->join(*newRange));
      }
    });

    return mlir::success();
  }
};

class IntegerRangeAnalysisEx : public mlir::dataflow::IntegerRangeAnalysis {
public:
  using IntegerRangeAnalysis::IntegerRangeAnalysis;

  void visitOperation(
      mlir::Operation *op,
      llvm::ArrayRef<const mlir::dataflow::IntegerValueRangeLattice *> operands,
      llvm::ArrayRef<mlir::dataflow::IntegerValueRangeLattice *> results)
      override {
    LLVM_DEBUG(llvm::dbgs() << "IntegerRangeAnalysisEx: Visiting operation: "
                            << *op << "\n");

    if (auto dim = mlir::dyn_cast<mlir::ShapedDimOpInterface>(op)) {
      assert(op->getNumResults() == 1);
      assert(results.size() == 1);

      auto *lattice = results.front();
      auto newRange = [&]() -> std::optional<mlir::ConstantIntRanges> {
        auto state =
            getOrCreateFor<ShapeValueLattice>(op, dim.getShapedValue());
        if (!state)
          return std::nullopt;

        auto &shapeVal = state->getValue();
        if (shapeVal.isUninitialized())
          return std::nullopt;

        auto index = mlir::getConstantIntValue(dim.getDimension());
        if (!index)
          return std::nullopt;

        auto shape = shapeVal.getShape();
        auto indexVal = *index;
        if (indexVal < 0 || indexVal >= static_cast<int64_t>(shape.size()))
          return std::nullopt;

        return shape[indexVal];
      }();

      if (newRange) {
        LLVM_DEBUG(llvm::dbgs() << "IntegerRangeAnalysisEx: New dim val: "
                                << newRange << "\n");

        auto changed = lattice->join(mlir::IntegerValueRange{newRange});
        propagateIfChanged(lattice, changed);
      }
      return;
    }

    if (auto select = mlir::dyn_cast<mlir::arith::SelectOp>(op)) {
      assert(op->getNumResults() == 1);
      assert(results.size() == 1);

      auto *lattice = results.front();
      auto newRange = [&]() {
        auto &condLattice = operands[0]->getValue();
        std::optional<mlir::APInt> mbCondVal =
            condLattice.isUninitialized()
                ? std::nullopt
                : condLattice.getValue().getConstantValue();

        const auto &trueCase = operands[1]->getValue();
        const auto &falseCase = operands[2]->getValue();

        if (mbCondVal) {
          if (mbCondVal->isZero())
            return falseCase;
          else
            return trueCase;
        }

        if (trueCase.isUninitialized() || falseCase.isUninitialized())
          return mlir::IntegerValueRange{};

        return mlir::IntegerValueRange::join(trueCase, falseCase);
      }();

      auto changed = lattice->join(mlir::IntegerValueRange{newRange});
      propagateIfChanged(lattice, changed);
      return;
    }

    // TODO: upstream
    if (!mlir::isa<mlir::InferIntRangeInterface>(op)) {
      for (auto &&[lattice, value] : llvm::zip(results, op->getResults())) {
        if (value.getType().isIntOrIndex()) {
          propagateIfChanged(
              lattice,
              lattice->join(mlir::IntegerValueRange::getMaxRange(value)));
        }
      }
      return setAllToEntryStates(results);
    }

    mlir::dataflow::IntegerRangeAnalysis::visitOperation(op, operands, results);
  }

  void visitNonControlFlowArguments(
      mlir::Operation *op, const mlir::RegionSuccessor &successor,
      mlir::ArrayRef<mlir::dataflow::IntegerValueRangeLattice *> argLattices,
      unsigned firstIndex) override {

    if (auto loop = mlir::dyn_cast<mlir::LoopLikeOpInterface>(op)) {
      if (op->getNumResults() == 0) {
        std::optional<mlir::Value> iv = loop.getSingleInductionVar();
        if (iv) {
          mlir::dataflow::IntegerValueRangeLattice *ivEntry =
              getLatticeElement(*iv);
          propagateIfChanged(
              ivEntry,
              ivEntry->join(mlir::IntegerValueRange::getMaxRange(*iv)));
          return;
        }
      }
    }

    mlir::dataflow::IntegerRangeAnalysis::visitNonControlFlowArguments(
        op, successor, argLattices, firstIndex);
  }
};

static void printShapeAnalysisState(mlir::DataFlowSolver &solver,
                                    mlir::Operation *root) {
  assert(root && "Invalid root");
  auto ctx = root->getContext();
  auto attrName = mlir::StringAttr::get(ctx, "int_range");
  auto addAttr = [&](mlir::Operation *op) {
    if (op->getNumResults() == 0)
      return;

    std::string str;
    llvm::raw_string_ostream os(str);
    for (auto &&[i, res] : llvm::enumerate(op->getResults())) {
      if (i != 0)
        os << ", ";

      auto *lattice =
          solver.lookupState<mlir::dataflow::IntegerValueRangeLattice>(res);
      if (!lattice || lattice->getValue().isUninitialized()) {
        os << "<none>";
        continue;
      }

      lattice->getValue().print(os);
    }
    os.flush();

    auto attr = mlir::StringAttr::get(ctx, str);
    op->setAttr(attrName, attr);
  };

  auto removeAttr = [&](mlir::Operation *op) { op->removeAttr(attrName); };

  root->walk(addAttr);
  llvm::dbgs() << *root;
  root->walk(removeAttr);
}

struct ShapeIntegerRangePropagationPass
    : public mlir::PassWrapper<ShapeIntegerRangePropagationPass,
                               mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ShapeIntegerRangePropagationPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::linalg::LinalgDialect>();
    registry.insert<mlir::tensor::TensorDialect>();
    registry.insert<numba::ntensor::NTensorDialect>();
    registry.insert<numba::util::NumbaUtilDialect>();
  }

  void runOnOperation() override {
    LLVM_DEBUG(llvm::dbgs() << "ShapeIntegerRangePropagationPass:\n");
    auto op = getOperation();
    mlir::DataFlowSolver solver;
    solver.load<mlir::dataflow::DeadCodeAnalysis>();
    solver.load<ShapeValueAnalysis>();
    solver.load<IntegerRangeAnalysisEx>();
    if (failed(solver.initializeAndRun(op)))
      return signalPassFailure();

    LLVM_DEBUG(printShapeAnalysisState(solver, op));

    auto *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);

    mlir::arith::populateIntRangeOptimizationsPatterns(patterns, solver);

    if (mlir::failed(
            mlir::applyPatternsAndFoldGreedily(op, std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<mlir::Pass> numba::createShapeIntegerRangePropagationPass() {
  return std::make_unique<ShapeIntegerRangePropagationPass>();
}
