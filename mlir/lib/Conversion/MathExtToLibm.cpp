// SPDX-FileCopyrightText: 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "numba/Conversion/MathExtToLibm.hpp"

#include "numba/Dialect/math_ext/IR/MathExt.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Utils/IndexingUtils.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

namespace {
// Pattern to convert vector operations to scalar operations. This is needed as
// libm calls require scalars.
template <typename Op>
struct VecOpToScalarOp : public mlir::OpRewritePattern<Op> {
public:
  using mlir::OpRewritePattern<Op>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(Op op, mlir::PatternRewriter &rewriter) const final {
    auto opType = op.getType();
    auto loc = op.getLoc();
    auto vecType = mlir::dyn_cast<mlir::VectorType>(opType);

    if (!vecType)
      return mlir::failure();
    if (!vecType.hasRank())
      return mlir::failure();
    auto shape = vecType.getShape();
    int64_t numElements = vecType.getNumElements();

    mlir::Value result = rewriter.create<mlir::arith::ConstantOp>(
        loc, mlir::DenseElementsAttr::get(
                 vecType, mlir::FloatAttr::get(vecType.getElementType(), 0.0)));
    mlir::SmallVector<int64_t> strides = mlir::computeStrides(shape);
    for (auto linearIndex = 0; linearIndex < numElements; ++linearIndex) {
      mlir::SmallVector<int64_t> positions =
          mlir::delinearize(linearIndex, strides);
      mlir::SmallVector<mlir::Value> operands;
      for (auto input : op->getOperands())
        operands.push_back(
            rewriter.create<mlir::vector::ExtractOp>(loc, input, positions));
      mlir::Value scalarOp =
          rewriter.create<Op>(loc, vecType.getElementType(), operands);
      result = rewriter.create<mlir::vector::InsertOp>(loc, scalarOp, result,
                                                       positions);
    }
    rewriter.replaceOp(op, {result});
    return mlir::success();
  }
};

template <typename Op>
struct PromoteOpToF32 : public mlir::OpRewritePattern<Op> {
public:
  using mlir::OpRewritePattern<Op>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(Op op, mlir::PatternRewriter &rewriter) const final {
    auto opType = op.getType();
    if (!mlir::isa<mlir::Float16Type, mlir::BFloat16Type>(opType))
      return mlir::failure();

    auto loc = op.getLoc();
    auto f32 = rewriter.getF32Type();
    auto extendedOperands = llvm::to_vector(llvm::map_range(
        op->getOperands(), [&](mlir::Value operand) -> mlir::Value {
          return rewriter.create<mlir::arith::ExtFOp>(loc, f32, operand);
        }));
    auto newOp = rewriter.create<Op>(loc, f32, extendedOperands);
    rewriter.replaceOpWithNewOp<mlir::arith::TruncFOp>(op, opType, newOp);
    return mlir::success();
  }
};

template <typename Op>
struct ScalarOpToLibmCall : public mlir::OpRewritePattern<Op> {
public:
  using mlir::OpRewritePattern<Op>::OpRewritePattern;
  ScalarOpToLibmCall(mlir::MLIRContext *context, mlir::StringRef floatFunc,
                     mlir::StringRef doubleFunc)
      : mlir::OpRewritePattern<Op>(context), floatFunc(floatFunc),
        doubleFunc(doubleFunc){};

  mlir::LogicalResult
  matchAndRewrite(Op op, mlir::PatternRewriter &rewriter) const final {
    auto module = mlir::SymbolTable::getNearestSymbolTable(op);
    auto type = op.getType();
    if (!mlir::isa<mlir::Float32Type, mlir::Float64Type>(type))
      return mlir::failure();

    auto name = type.getIntOrFloatBitWidth() == 64 ? doubleFunc : floatFunc;
    auto opFunc = mlir::dyn_cast_or_null<mlir::SymbolOpInterface>(
        mlir::SymbolTable::lookupSymbolIn(module, name));
    // Forward declare function if it hasn't already been
    if (!opFunc) {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&module->getRegion(0).front());
      auto opFunctionTy = mlir::FunctionType::get(
          rewriter.getContext(), op->getOperandTypes(), op->getResultTypes());
      opFunc = rewriter.create<mlir::func::FuncOp>(rewriter.getUnknownLoc(),
                                                   name, opFunctionTy);
      opFunc.setPrivate();

      // By definition Math dialect operations imply LLVM's "readnone"
      // function attribute, so we can set it here to provide more
      // optimization opportunities (e.g. LICM) for backends targeting LLVM IR.
      // This will have to be changed, when strict FP behavior is supported
      // by Math dialect.
      opFunc->setAttr(mlir::LLVM::LLVMDialect::getReadnoneAttrName(),
                      mlir::UnitAttr::get(rewriter.getContext()));
    }
    assert(mlir::isa<mlir::FunctionOpInterface>(
        mlir::SymbolTable::lookupSymbolIn(module, name)));

    rewriter.replaceOpWithNewOp<mlir::func::CallOp>(op, name, op.getType(),
                                                    op->getOperands());

    return mlir::success();
  }

private:
  std::string floatFunc, doubleFunc;
};

template <typename OpTy>
static void
populatePatternsForOp(mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx,
                      mlir::StringRef floatFunc, mlir::StringRef doubleFunc) {
  patterns.add<VecOpToScalarOp<OpTy>, PromoteOpToF32<OpTy>>(ctx);
  patterns.add<ScalarOpToLibmCall<OpTy>>(ctx, floatFunc, doubleFunc);
}

struct MathExtToLibmPass
    : public mlir::PassWrapper<MathExtToLibmPass, mlir::OperationPass<void>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MathExtToLibmPass)

  void runOnOperation() override {
    auto op = getOperation();

    mlir::RewritePatternSet patterns(&getContext());
    numba::populateMathExtToLibmPatterns(patterns);

    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<mlir::arith::ArithDialect, mlir::BuiltinDialect,
                           mlir::func::FuncDialect,
                           mlir::vector::VectorDialect>();
    target.addIllegalDialect<numba::math_ext::MathExtDialect>();
    if (failed(applyPartialConversion(op, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

void numba::populateMathExtToLibmPatterns(mlir::RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();

  populatePatternsForOp<numba::math_ext::AcosOp>(patterns, ctx, "acosf",
                                                 "acos");
}

std::unique_ptr<mlir::Pass> numba::createMathExtToLibmPass() {
  return std::make_unique<MathExtToLibmPass>();
}
