// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#define _USE_MATH_DEFINES
#include <cmath>

#include "pipelines/PlierToScf.hpp"
#include "pipelines/PlierToStd.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Complex/IR/Complex.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/Passes.h>

#include "numba/Dialect/numba_util/Dialect.hpp"
#include "numba/Dialect/plier/Dialect.hpp"

#include "numba/Compiler/PipelineRegistry.hpp"
#include "numba/Transforms/CallLowering.hpp"
#include "numba/Transforms/CastUtils.hpp"
#include "numba/Transforms/ConstUtils.hpp"
#include "numba/Transforms/InlineUtils.hpp"
#include "numba/Transforms/PipelineUtils.hpp"
#include "numba/Transforms/PromoteToParallel.hpp"
#include "numba/Transforms/RewriteWrapper.hpp"
#include "numba/Transforms/TypeConversion.hpp"

#include "BasePipeline.hpp"
#include "PyFuncResolver.hpp"
#include "PyLinalgResolver.hpp"

namespace {
static bool isSupportedType(mlir::Type type) {
  assert(type);
  return type.isa<mlir::IntegerType, mlir::FloatType, mlir::ComplexType>();
}

static bool isSupportedConst(mlir::Type type) {
  assert(type);
  return type.isa<mlir::IntegerType, mlir::FloatType, mlir::ComplexType,
                  mlir::TupleType, mlir::NoneType, numba::util::TypeVarType,
                  numba::util::StringType>();
}

static bool isInt(mlir::Type type) {
  assert(type);
  return type.isa<mlir::IntegerType>();
}

static bool isFloat(mlir::Type type) {
  assert(type);
  return type.isa<mlir::FloatType>();
}

static bool isComplex(mlir::Type type) {
  assert(type);
  return type.isa<mlir::ComplexType>();
}

static mlir::Value lowerConst(mlir::OpBuilder &builder, mlir::Location loc,
                              mlir::Type type, mlir::Attribute attr) {
  if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr)) {
    auto intVal = intAttr.getValue().getSExtValue();
    auto origType = mlir::cast<mlir::IntegerType>(intAttr.getType());
    auto constType = numba::makeSignlessType(origType);
    mlir::Value res = builder.create<mlir::arith::ConstantIntOp>(
        loc, intVal, constType.getWidth());
    return numba::doConvert(builder, loc, res, type);
  }

  if (auto floatAttr = mlir::dyn_cast<mlir::FloatAttr>(attr)) {
    mlir::Value res = builder.create<mlir::arith::ConstantOp>(loc, floatAttr);
    return numba::doConvert(builder, loc, res, type);
  }

  if (auto complexAttr = mlir::dyn_cast<mlir::complex::NumberAttr>(attr)) {
    const double vals[] = {
        complexAttr.getReal().convertToDouble(),
        complexAttr.getImag().convertToDouble(),
    };
    auto arr = builder.getF64ArrayAttr(vals);
    mlir::Value res = builder.create<mlir::complex::ConstantOp>(
        loc, complexAttr.getType(), arr);
    return numba::doConvert(builder, loc, res, type);
  }

  if (auto array = mlir::dyn_cast<mlir::ArrayAttr>(attr)) {
    auto tupleType = mlir::dyn_cast<mlir::TupleType>(type);
    if (!tupleType || tupleType.size() != array.size())
      return nullptr;

    llvm::SmallVector<mlir::Value> values(array.size());
    for (auto &&[i, elemAttr] : llvm::enumerate(array)) {
      auto val = lowerConst(builder, loc, tupleType.getType(i), elemAttr);
      if (!val)
        return nullptr;

      values[i] = val;
    }

    mlir::ValueRange valRange(values);
    auto retType = builder.getTupleType(valRange);
    return builder.create<numba::util::BuildTupleOp>(loc, retType, values)
        .getResult();
  }

  if (auto string = mlir::dyn_cast<mlir::StringAttr>(attr)) {
    if (!mlir::isa<numba::util::StringType>(type))
      return nullptr;

    return builder.create<numba::util::StringConstOp>(loc, string);
  }

  return nullptr;
}

struct UndefOpLowering : public mlir::OpConversionPattern<plier::UndefOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::UndefOp op, plier::UndefOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto &converter = *getTypeConverter();
    auto expectedType = converter.convertType(op.getType());
    if (!expectedType)
      return mlir::failure();

    expectedType = numba::makeSignlessType(expectedType);
    rewriter.replaceOpWithNewOp<mlir::ub::PoisonOp>(op, expectedType, nullptr);

    return mlir::success();
  }
};

struct ConstOpLowering : public mlir::OpConversionPattern<plier::ConstOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::ConstOp op, plier::ConstOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto &converter = *getTypeConverter();
    auto expectedType = converter.convertType(op.getType());
    if (!expectedType)
      return mlir::failure();

    if (mlir::isa<mlir::NoneType>(expectedType)) {
      rewriter.replaceOpWithNewOp<mlir::ub::PoisonOp>(op, expectedType,
                                                      nullptr);

      return mlir::success();
    }

    auto value =
        lowerConst(rewriter, op.getLoc(), expectedType, adaptor.getValAttr());
    if (!value)
      return mlir::failure();

    rewriter.replaceOp(op, value);
    return mlir::success();
  }
};

static bool isOmittedType(mlir::Type type) {
  return type.isa<plier::OmittedType>();
}

static mlir::TypedAttr makeSignlessAttr(mlir::TypedAttr val) {
  auto type = val.cast<mlir::TypedAttr>().getType();
  if (auto intType = type.dyn_cast<mlir::IntegerType>()) {
    if (!intType.isSignless()) {
      auto newType = numba::makeSignlessType(intType);
      return mlir::IntegerAttr::get(
          newType, numba::getIntAttrValue(val.cast<mlir::IntegerAttr>()));
    }
  }
  return val;
}

template <typename Op>
struct LiteralLowering : public mlir::OpConversionPattern<Op> {
  using mlir::OpConversionPattern<Op>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto type = op.getType();
    auto &converter = *(this->getTypeConverter());
    auto convertedType = converter.convertType(type);
    if (!convertedType)
      return mlir::failure();

    if (mlir::isa<mlir::NoneType>(convertedType)) {
      rewriter.replaceOpWithNewOp<mlir::ub::PoisonOp>(op, convertedType,
                                                      nullptr);
      return mlir::success();
    }

    if (auto typevar =
            mlir::dyn_cast<numba::util::TypeVarType>(convertedType)) {
      rewriter.replaceOpWithNewOp<mlir::ub::PoisonOp>(op, typevar, nullptr);
      return mlir::success();
    }

    return mlir::failure();
  }
};

struct OmittedLowering : public mlir::OpConversionPattern<plier::CastOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::CastOp op, plier::CastOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto type = op.getType();
    auto &converter = *(this->getTypeConverter());
    auto convertedType = converter.convertType(type);
    if (!convertedType)
      return mlir::failure();

    auto getOmittedValue = [&](mlir::Type type,
                               mlir::Type dstType) -> mlir::TypedAttr {
      if (auto attr = type.dyn_cast<plier::OmittedType>())
        return mlir::cast<mlir::TypedAttr>(attr.getValue());

      return {};
    };

    if (auto omittedAttr =
            getOmittedValue(adaptor.getValue().getType(), convertedType)) {
      auto loc = op.getLoc();
      auto dstType = omittedAttr.cast<mlir::TypedAttr>().getType();
      auto val = makeSignlessAttr(omittedAttr);
      auto newVal =
          rewriter.create<mlir::arith::ConstantOp>(loc, val).getResult();
      if (dstType != val.cast<mlir::TypedAttr>().getType())
        newVal = rewriter.create<numba::util::SignCastOp>(loc, dstType, newVal);

      rewriter.replaceOp(op, newVal);
      return mlir::success();
    }
    return mlir::failure();
  }
};

static unsigned getBitsCount(mlir::Type type) {
  assert(type);
  if (type.isa<mlir::IntegerType>())
    return type.cast<mlir::IntegerType>().getWidth();

  if (type.isa<mlir::Float16Type>())
    return 11;

  if (type.isa<mlir::Float32Type>())
    return 24;

  if (type.isa<mlir::Float64Type>())
    return 53;

  if (auto c = type.dyn_cast<mlir::ComplexType>()) {
    return getBitsCount(c.getElementType());
  }

  llvm_unreachable("Unhandled type");
};

static mlir::Type coerce(mlir::Type type0, mlir::Type type1) {
  if (type0 == type1)
    return type0;

  auto c0 = isComplex(type0);
  auto c1 = isComplex(type1);
  if (c0 && !c1)
    return type0;

  if (!c0 && c1)
    return type1;

  auto f0 = isFloat(type0);
  auto f1 = isFloat(type1);
  if (f0 && !f1)
    return type0;

  if (!f0 && f1)
    return type1;

  return getBitsCount(type0) < getBitsCount(type1) ? type1 : type0;
}

static mlir::Value invalidReplaceOp(mlir::PatternRewriter & /*rewriter*/,
                                    mlir::Location /*loc*/,
                                    mlir::ValueRange /*operands*/,
                                    mlir::Type /*newType*/) {
  llvm_unreachable("invalidReplaceOp");
}

template <typename T>
static mlir::Value replaceOp(mlir::PatternRewriter &rewriter,
                             mlir::Location loc, mlir::ValueRange operands,
                             mlir::Type newType) {
  auto signlessType = numba::makeSignlessType(newType);
  llvm::SmallVector<mlir::Value> newOperands(operands.size());
  for (auto &&[i, val] : llvm::enumerate(operands))
    newOperands[i] = numba::doConvert(rewriter, loc, val, signlessType);

  auto res = rewriter.createOrFold<T>(loc, newOperands);
  return numba::doConvert(rewriter, loc, res, newType);
}

static mlir::Value replaceRShiftOp(mlir::PatternRewriter &rewriter,
                                   mlir::Location loc,
                                   mlir::ValueRange operands,
                                   mlir::Type newType) {
  auto isUnsigned = newType.isUnsignedInteger();
  auto signlessType = numba::makeSignlessType(newType);
  llvm::SmallVector<mlir::Value> newOperands(operands.size());
  for (auto &&[i, val] : llvm::enumerate(operands))
    newOperands[i] = numba::doConvert(rewriter, loc, val, signlessType);

  mlir::Value res;
  if (isUnsigned) {
    res = rewriter.createOrFold<mlir::arith::ShRUIOp>(loc, newOperands);
  } else {
    res = rewriter.createOrFold<mlir::arith::ShRSIOp>(loc, newOperands);
  }
  return numba::doConvert(rewriter, loc, res, newType);
}

mlir::Value replaceIpowOp(mlir::PatternRewriter &rewriter, mlir::Location loc,
                          mlir::ValueRange operands, mlir::Type newType) {
  auto f64Type = rewriter.getF64Type();
  auto a = numba::doConvert(rewriter, loc, operands[0], f64Type);
  auto b = numba::doConvert(rewriter, loc, operands[1], f64Type);
  auto fres = rewriter.create<mlir::math::PowFOp>(loc, a, b).getResult();
  return numba::doConvert(rewriter, loc, fres, newType);
}

mlir::Value replaceItruedivOp(mlir::PatternRewriter &rewriter,
                              mlir::Location loc, mlir::ValueRange operands,
                              mlir::Type newType) {
  assert(newType.isa<mlir::FloatType>());
  auto lhs = numba::doConvert(rewriter, loc, operands[0], newType);
  auto rhs = numba::doConvert(rewriter, loc, operands[1], newType);
  return rewriter.createOrFold<mlir::arith::DivFOp>(loc, lhs, rhs);
}

mlir::Value replaceIfloordivOp(mlir::PatternRewriter &rewriter,
                               mlir::Location loc, mlir::ValueRange operands,
                               mlir::Type newType) {
  auto newIntType = newType.cast<mlir::IntegerType>();
  auto signlessType = numba::makeSignlessType(newIntType);
  auto lhs = numba::doConvert(rewriter, loc, operands[0], signlessType);
  auto rhs = numba::doConvert(rewriter, loc, operands[1], signlessType);
  mlir::Value res;
  if (newIntType.isSigned()) {
    res = rewriter.createOrFold<mlir::arith::FloorDivSIOp>(loc, lhs, rhs);
  } else {
    res = rewriter.createOrFold<mlir::arith::DivUIOp>(loc, lhs, rhs);
  }
  return numba::doConvert(rewriter, loc, res, newType);
}

mlir::Value replaceFfloordivOp(mlir::PatternRewriter &rewriter,
                               mlir::Location loc, mlir::ValueRange operands,
                               mlir::Type newType) {
  assert(newType.isa<mlir::FloatType>());
  auto lhs = numba::doConvert(rewriter, loc, operands[0], newType);
  auto rhs = numba::doConvert(rewriter, loc, operands[1], newType);
  auto res = rewriter.createOrFold<mlir::arith::DivFOp>(loc, lhs, rhs);
  return rewriter.createOrFold<mlir::math::FloorOp>(loc, res);
}

mlir::Value replaceImodOp(mlir::PatternRewriter &rewriter, mlir::Location loc,
                          mlir::ValueRange operands, mlir::Type newType) {
  auto signlessType = numba::makeSignlessType(operands[0].getType());
  auto a = numba::doConvert(rewriter, loc, operands[0], signlessType);
  auto b = numba::doConvert(rewriter, loc, operands[1], signlessType);
  auto v1 = rewriter.create<mlir::arith::RemSIOp>(loc, a, b).getResult();
  auto v2 = rewriter.create<mlir::arith::AddIOp>(loc, v1, b).getResult();
  auto res = rewriter.create<mlir::arith::RemSIOp>(loc, v2, b).getResult();
  return numba::doConvert(rewriter, loc, res, newType);
}

mlir::Value replaceFmodOp(mlir::PatternRewriter &rewriter, mlir::Location loc,
                          mlir::ValueRange operands, mlir::Type /*newType*/) {
  auto a = operands[0];
  auto b = operands[1];
  auto v1 = rewriter.create<mlir::arith::RemFOp>(loc, a, b).getResult();
  auto v2 = rewriter.create<mlir::arith::AddFOp>(loc, v1, b).getResult();
  return rewriter.create<mlir::arith::RemFOp>(loc, v2, b).getResult();
}

template <mlir::arith::CmpIPredicate SignedPred,
          mlir::arith::CmpIPredicate UnsignedPred = SignedPred>
mlir::Value replaceCmpiOp(mlir::PatternRewriter &rewriter, mlir::Location loc,
                          mlir::ValueRange operands, mlir::Type /*newType*/) {
  assert(operands.size() == 2);
  assert(operands[0].getType() == operands[1].getType());
  auto type = operands[0].getType().cast<mlir::IntegerType>();
  auto signlessType = numba::makeSignlessType(type);
  auto a = numba::doConvert(rewriter, loc, operands[0], signlessType);
  auto b = numba::doConvert(rewriter, loc, operands[1], signlessType);
  if (SignedPred == UnsignedPred || type.isSigned()) {
    return rewriter.createOrFold<mlir::arith::CmpIOp>(loc, SignedPred, a, b);
  } else {
    return rewriter.createOrFold<mlir::arith::CmpIOp>(loc, UnsignedPred, a, b);
  }
}

template <mlir::arith::CmpFPredicate Pred>
mlir::Value replaceCmpfOp(mlir::PatternRewriter &rewriter, mlir::Location loc,
                          mlir::ValueRange operands, mlir::Type /*newType*/) {
  auto signlessType = numba::makeSignlessType(operands[0].getType());
  auto a = numba::doConvert(rewriter, loc, operands[0], signlessType);
  auto b = numba::doConvert(rewriter, loc, operands[1], signlessType);
  return rewriter.createOrFold<mlir::arith::CmpFOp>(loc, Pred, a, b);
}

struct InplaceBinOpLowering
    : public mlir::OpConversionPattern<plier::InplaceBinOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::InplaceBinOp op, plier::InplaceBinOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    assert(converter && "Invalid type converter");

    auto resType = converter->convertType(op.getResult().getType());
    if (!resType)
      return mlir::failure();

    rewriter.replaceOpWithNewOp<plier::BinOp>(
        op, resType, adaptor.getLhs(), adaptor.getRhs(), adaptor.getOpAttr());
    return mlir::success();
  }
};

struct BinOpLowering : public mlir::OpConversionPattern<plier::BinOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::BinOp op, plier::BinOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto &converter = *getTypeConverter();
    auto operands = adaptor.getOperands();
    assert(operands.size() == 2);
    auto type0 = operands[0].getType();
    auto type1 = operands[1].getType();
    if (!isSupportedType(type0) || !isSupportedType(type1))
      return mlir::failure();

    auto resType = converter.convertType(op.getType());
    if (!resType || !isSupportedType(resType))
      return mlir::failure();

    auto loc = op.getLoc();
    auto literalCast = [&](mlir::Value val, mlir::Type dstType) -> mlir::Value {
      return numba::doConvert(rewriter, loc, val, dstType);
    };

    std::array<mlir::Value, 2> convertedOperands = {
        literalCast(operands[0], type0), literalCast(operands[1], type1)};
    auto finalType = coerce(coerce(type0, type1), resType);
    assert(finalType);

    convertedOperands = {
        numba::doConvert(rewriter, loc, convertedOperands[0], finalType),
        numba::doConvert(rewriter, loc, convertedOperands[1], finalType)};

    assert(convertedOperands[0]);
    assert(convertedOperands[1]);

    using func_t = mlir::Value (*)(mlir::PatternRewriter &, mlir::Location,
                                   mlir::ValueRange, mlir::Type);
    struct OpDesc {
      llvm::StringRef type;
      func_t iop;
      func_t fop;
      func_t cop;
    };

    const OpDesc handlers[] = {
        {"+", &replaceOp<mlir::arith::AddIOp>, &replaceOp<mlir::arith::AddFOp>,
         &replaceOp<mlir::complex::AddOp>},
        {"-", &replaceOp<mlir::arith::SubIOp>, &replaceOp<mlir::arith::SubFOp>,
         &replaceOp<mlir::complex::SubOp>},
        {"*", &replaceOp<mlir::arith::MulIOp>, &replaceOp<mlir::arith::MulFOp>,
         &replaceOp<mlir::complex::MulOp>},
        {"**", &replaceIpowOp, &replaceOp<mlir::math::PowFOp>,
         &replaceOp<mlir::complex::PowOp>},
        {"/", &replaceItruedivOp, &replaceOp<mlir::arith::DivFOp>,
         &replaceOp<mlir::complex::DivOp>},
        {"//", &replaceIfloordivOp, &replaceFfloordivOp, &invalidReplaceOp},
        {"%", &replaceImodOp, &replaceFmodOp, &invalidReplaceOp},
        {"&", &replaceOp<mlir::arith::AndIOp>, &invalidReplaceOp,
         &invalidReplaceOp},
        {"|", &replaceOp<mlir::arith::OrIOp>, &invalidReplaceOp,
         &invalidReplaceOp},
        {"^", &replaceOp<mlir::arith::XOrIOp>, &invalidReplaceOp,
         &invalidReplaceOp},
        {">>", &replaceRShiftOp, &invalidReplaceOp, &invalidReplaceOp},
        {"<<", &replaceOp<mlir::arith::ShLIOp>, &invalidReplaceOp,
         &invalidReplaceOp},

        {">",
         &replaceCmpiOp<mlir::arith::CmpIPredicate::sgt,
                        mlir::arith::CmpIPredicate::ugt>,
         &replaceCmpfOp<mlir::arith::CmpFPredicate::OGT>, &invalidReplaceOp},
        {">=",
         &replaceCmpiOp<mlir::arith::CmpIPredicate::sge,
                        mlir::arith::CmpIPredicate::uge>,
         &replaceCmpfOp<mlir::arith::CmpFPredicate::OGE>, &invalidReplaceOp},
        {"<",
         &replaceCmpiOp<mlir::arith::CmpIPredicate::slt,
                        mlir::arith::CmpIPredicate::ult>,
         &replaceCmpfOp<mlir::arith::CmpFPredicate::OLT>, &invalidReplaceOp},
        {"<=",
         &replaceCmpiOp<mlir::arith::CmpIPredicate::sle,
                        mlir::arith::CmpIPredicate::ule>,
         &replaceCmpfOp<mlir::arith::CmpFPredicate::OLE>, &invalidReplaceOp},
        {"!=", &replaceCmpiOp<mlir::arith::CmpIPredicate::ne>,
         &replaceCmpfOp<mlir::arith::CmpFPredicate::ONE>, &invalidReplaceOp},
        {"==", &replaceCmpiOp<mlir::arith::CmpIPredicate::eq>,
         &replaceCmpfOp<mlir::arith::CmpFPredicate::OEQ>, &invalidReplaceOp},
    };

    using membptr_t = func_t OpDesc::*;
    auto callHandler = [&](membptr_t mem) {
      for (auto &h : handlers) {
        if (h.type == op.getOp()) {
          auto res = (h.*mem)(rewriter, loc, convertedOperands, finalType);
          if (res.getType() != resType)
            res = numba::doConvert(rewriter, loc, res, resType);

          rewriter.replaceOp(op, res);
          return mlir::success();
        }
      }
      return mlir::failure();
    };

    if (isInt(finalType)) {
      return callHandler(&OpDesc::iop);
    } else if (isFloat(finalType)) {
      return callHandler(&OpDesc::fop);
    } else if (isComplex(finalType)) {
      return callHandler(&OpDesc::cop);
    }
    return mlir::failure();
  }
};

struct BinOpTupleLowering : public mlir::OpConversionPattern<plier::BinOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::BinOp op, plier::BinOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    auto lhsType = lhs.getType().dyn_cast<mlir::TupleType>();
    if (!lhsType)
      return mlir::failure();

    auto loc = op->getLoc();
    if (adaptor.getOp() == "+") {
      auto rhsType = rhs.getType().dyn_cast<mlir::TupleType>();
      if (!rhsType)
        return mlir::failure();

      auto count = lhsType.size() + rhsType.size();
      llvm::SmallVector<mlir::Value> newArgs;
      llvm::SmallVector<mlir::Type> newTypes;
      newArgs.reserve(count);
      newTypes.reserve(count);

      for (auto &arg : {lhs, rhs}) {
        auto type = arg.getType().cast<mlir::TupleType>();
        for (auto i : llvm::seq<size_t>(0, type.size())) {
          auto elemType = type.getType(i);
          auto ind = rewriter.create<mlir::arith::ConstantIndexOp>(
              loc, static_cast<int64_t>(i));
          auto elem = rewriter.create<numba::util::TupleExtractOp>(
              loc, elemType, arg, ind);
          newArgs.emplace_back(elem);
          newTypes.emplace_back(elemType);
        }
      }

      auto newTupleType = mlir::TupleType::get(getContext(), newTypes);
      rewriter.replaceOpWithNewOp<numba::util::BuildTupleOp>(op, newTupleType,
                                                             newArgs);
      return mlir::success();
    }

    return mlir::failure();
  }
};

static mlir::Value negate(mlir::PatternRewriter &rewriter, mlir::Location loc,
                          mlir::Value val, mlir::Type resType) {
  val = numba::doConvert(rewriter, loc, val, resType);
  if (auto itype = resType.dyn_cast<mlir::IntegerType>()) {
    auto signless = numba::makeSignlessType(resType);
    if (signless != itype)
      val = rewriter.create<numba::util::SignCastOp>(loc, signless, val);

    // TODO: no int negation?
    auto zero = rewriter.create<mlir::arith::ConstantOp>(
        loc, mlir::IntegerAttr::get(signless, 0));
    auto res = rewriter.create<mlir::arith::SubIOp>(loc, zero, val).getResult();
    if (signless != itype)
      res = rewriter.create<numba::util::SignCastOp>(loc, itype, res);

    return res;
  }

  if (resType.isa<mlir::FloatType>())
    return rewriter.create<mlir::arith::NegFOp>(loc, val);

  if (resType.isa<mlir::ComplexType>())
    return rewriter.create<mlir::complex::NegOp>(loc, val);

  llvm_unreachable("negate: unsupported type");
}

static mlir::Value unaryPlus(mlir::PatternRewriter &rewriter,
                             mlir::Location loc, mlir::Value arg,
                             mlir::Type resType) {
  return numba::doConvert(rewriter, loc, arg, resType);
}

static mlir::Value unaryMinus(mlir::PatternRewriter &rewriter,
                              mlir::Location loc, mlir::Value arg,
                              mlir::Type resType) {
  return negate(rewriter, loc, arg, resType);
}

static mlir::Value unaryNot(mlir::PatternRewriter &rewriter, mlir::Location loc,
                            mlir::Value arg, mlir::Type resType) {
  auto i1 = rewriter.getIntegerType(1);
  auto casted = numba::doConvert(rewriter, loc, arg, i1);
  auto one = rewriter.create<mlir::arith::ConstantIntOp>(loc, 1, i1);
  return rewriter.create<mlir::arith::SubIOp>(loc, one, casted);
}

static mlir::Value unaryInvert(mlir::PatternRewriter &rewriter,
                               mlir::Location loc, mlir::Value arg,
                               mlir::Type resType) {
  auto intType = arg.getType().dyn_cast<mlir::IntegerType>();
  if (!intType)
    return {};

  mlir::Type signlessType;
  if (intType.getWidth() == 1) {
    intType = rewriter.getIntegerType(64);
    signlessType = intType;
    arg = rewriter.create<mlir::arith::ExtUIOp>(loc, intType, arg);
  } else {
    signlessType = numba::makeSignlessType(intType);
    if (intType != signlessType)
      arg = rewriter.create<numba::util::SignCastOp>(loc, signlessType, arg);
  }

  auto all = rewriter.create<mlir::arith::ConstantIntOp>(loc, -1, signlessType);

  arg = rewriter.create<mlir::arith::XOrIOp>(loc, all, arg);

  if (intType != signlessType)
    arg = rewriter.create<numba::util::SignCastOp>(loc, intType, arg);

  if (resType != arg.getType())
    arg = numba::doConvert(rewriter, loc, arg, resType);

  return arg;
}

struct UnaryOpLowering : public mlir::OpConversionPattern<plier::UnaryOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::UnaryOp op, plier::UnaryOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto &converter = *getTypeConverter();
    auto arg = adaptor.getValue();
    auto type = arg.getType();
    if (!isSupportedType(type))
      return mlir::failure();

    auto resType = converter.convertType(op.getType());
    if (!resType)
      return mlir::failure();

    using func_t = mlir::Value (*)(mlir::PatternRewriter &, mlir::Location,
                                   mlir::Value, mlir::Type);

    using Handler = std::pair<llvm::StringRef, func_t>;
    const Handler handlers[] = {
        {"+", &unaryPlus},
        {"-", &unaryMinus},
        {"not", &unaryNot},
        {"~", &unaryInvert},
    };

    auto opname = op.getOp();
    for (auto &h : handlers) {
      if (h.first == opname) {
        auto loc = op.getLoc();
        auto res = h.second(rewriter, loc, arg, resType);
        if (!res)
          return mlir::failure();

        rewriter.replaceOp(op, res);
        return mlir::success();
      }
    }

    return mlir::failure();
  }
};

struct LowerCasts : public mlir::OpConversionPattern<plier::CastOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::CastOp op, plier::CastOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto &converter = *getTypeConverter();
    auto src = adaptor.getValue();
    auto dstType = converter.convertType(op.getType());
    if (!dstType)
      return mlir::failure();

    auto srcType = src.getType();
    if (srcType == dstType) {
      rewriter.replaceOpWithNewOp<mlir::UnrealizedConversionCastOp>(op, dstType,
                                                                    src);
      return mlir::success();
    }

    auto res = numba::doConvert(rewriter, op.getLoc(), src, dstType);
    if (!res)
      return mlir::failure();

    rewriter.replaceOp(op, res);

    return mlir::success();
  }
};

static void rerunScfPipeline(mlir::Operation *op) {
  assert(nullptr != op);
  auto marker =
      mlir::StringAttr::get(op->getContext(), plierToScfPipelineName());
  auto mod = op->getParentOfType<mlir::ModuleOp>();
  assert(nullptr != mod);
  numba::addPipelineJumpMarker(mod, marker);
}

static mlir::LogicalResult
lowerSlice(plier::PyCallOp op, mlir::ValueRange operands,
           llvm::ArrayRef<std::pair<llvm::StringRef, mlir::Value>> kwargs,
           mlir::PatternRewriter &rewriter) {
  if (!kwargs.empty())
    return mlir::failure();

  if (operands.size() != 2 && operands.size() != 3)
    return mlir::failure();

  if (llvm::any_of(operands, [](mlir::Value op) {
        return !op.getType()
                    .isa<mlir::IntegerType, mlir::IndexType, mlir::NoneType>();
      }))
    return mlir::failure();

  auto begin = operands[0];
  auto end = operands[1];
  auto stride = [&]() -> mlir::Value {
    if (operands.size() == 3)
      return operands[2];

    return rewriter.create<mlir::arith::ConstantIndexOp>(op.getLoc(), 1);
  }();

  rerunScfPipeline(op);
  rewriter.replaceOpWithNewOp<plier::BuildSliceOp>(op, begin, end, stride);
  return mlir::success();
}

using kwargs_t = llvm::ArrayRef<std::pair<llvm::StringRef, mlir::Value>>;
using func_t = mlir::LogicalResult (*)(plier::PyCallOp, mlir::ValueRange,
                                       kwargs_t, mlir::PatternRewriter &);
static const std::pair<llvm::StringRef, func_t> builtinFuncsHandlers[] = {
    // clang-format off
    {"slice", &lowerSlice},
    // clang-format on
};

struct BuiltinCallsLowering final
    : public mlir::OpRewritePattern<plier::PyCallOp> {
  BuiltinCallsLowering(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<plier::PyCallOp>(context),
        resolver("numba_mlir.mlir.builtin.funcs", "registry") {}

  mlir::LogicalResult
  matchAndRewrite(plier::PyCallOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op.getVarargs())
      return mlir::failure();

    auto func = op.getFunc();
    if (!func || !mlir::isa<plier::FunctionType, numba::util::TypeVarType>(
                     func.getType()))
      return mlir::failure();

    auto funcName = op.getFuncName();

    llvm::SmallVector<std::pair<llvm::StringRef, mlir::Value>> kwargs;
    for (auto it : llvm::zip(op.getKwargs(), op.getKwNames())) {
      auto arg = std::get<0>(it);
      auto name = std::get<1>(it).cast<mlir::StringAttr>();
      kwargs.emplace_back(name.getValue(), arg);
    }

    auto loc = op.getLoc();
    return resolveCall(op, funcName, loc, rewriter, op.getArgs(), kwargs);
  }

protected:
  using KWargs = llvm::ArrayRef<std::pair<llvm::StringRef, mlir::Value>>;

  mlir::LogicalResult resolveCall(plier::PyCallOp op, mlir::StringRef name,
                                  mlir::Location loc,
                                  mlir::PatternRewriter &rewriter,
                                  mlir::ValueRange args, KWargs kwargs) const {
    for (auto &handler : builtinFuncsHandlers)
      if (handler.first == name)
        return handler.second(op, args, kwargs, rewriter);

    auto res = resolver.rewriteFunc(name, loc, rewriter, args, kwargs);
    if (!res)
      return mlir::failure();

    auto results = std::move(res).value();
    assert(results.size() == op->getNumResults());
    for (auto &&[i, r] : llvm::enumerate(results)) {
      auto dstType = op->getResultTypes()[i];
      if (dstType != r.getType())
        results[i] = rewriter.create<plier::CastOp>(loc, dstType, r);
    }

    rerunScfPipeline(op);
    rewriter.replaceOp(op, results);
    return mlir::success();
  }

private:
  PyLinalgResolver resolver;
};

static std::optional<mlir::Value> doCast(mlir::OpBuilder &builder,
                                         mlir::Location loc, mlir::Value src,
                                         mlir::Type dstType) {
  auto srcType = src.getType();
  if (srcType == dstType)
    return src;

  if (numba::canConvert(srcType, dstType))
    return numba::doConvert(builder, loc, src, dstType);

  return std::nullopt;
}

static mlir::Value doSafeCast(mlir::OpBuilder &builder, mlir::Location loc,
                              mlir::Value src, mlir::Type dstType) {
  auto res = doCast(builder, loc, src, dstType);
  if (res)
    return *res;

  return builder.create<plier::CastOp>(loc, dstType, src);
}

struct ExternalCallsLowering final : mlir::OpRewritePattern<plier::PyCallOp> {
  using OpRewritePattern::OpRewritePattern;

protected:
  mlir::LogicalResult
  matchAndRewrite(plier::PyCallOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto name = op.getFuncName();
    auto kwn = op.getKwNames();
    llvm::SmallVector<llvm::StringRef> kwNames;
    kwNames.reserve(kwn.size());
    for (auto name : kwn.getAsValueRange<mlir::StringAttr>())
      kwNames.emplace_back(name);

    auto mod = op->getParentOfType<mlir::ModuleOp>();
    assert(mod);

    auto loc = op.getLoc();
    auto res = resolver.getFunc(rewriter, loc, mod, name, op.getArgs(), kwNames,
                                op.getKwargs());
    if (!res)
      return mlir::failure();

    auto externalFunc = res->func;
    mlir::ValueRange mappedArgs(res->mappedArgs);

    llvm::SmallVector<mlir::Value> castedArgs(mappedArgs.size());
    auto funcTypes = externalFunc.getFunctionType().getInputs();
    for (auto &&[i, arg] : llvm::enumerate(mappedArgs)) {
      auto dstType = funcTypes[i];
      castedArgs[i] = doSafeCast(rewriter, loc, arg, dstType);
    }

    auto newFuncCall =
        rewriter.create<mlir::func::CallOp>(loc, externalFunc, castedArgs);

    auto results = newFuncCall.getResults();
    llvm::SmallVector<mlir::Value> castedResults(results.size());

    for (auto &&[ind, res] : llvm::enumerate(results)) {
      auto i = static_cast<unsigned>(ind);
      auto oldResType = op->getResult(i).getType();
      castedResults[i] = doSafeCast(rewriter, loc, res, oldResType);
    }

    rerunScfPipeline(op);
    rewriter.replaceOp(op, castedResults);
    return mlir::success();
  }

private:
  PyFuncResolver resolver;
};

struct BuiltinCallsLoweringPass
    : public numba::RewriteWrapperPass<
          BuiltinCallsLoweringPass, void, void, BuiltinCallsLowering,
          numba::ExpandCallVarargs, ExternalCallsLowering> {};

struct PlierToStdPass
    : public mlir::PassWrapper<PlierToStdPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PlierToStdPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::complex::ComplexDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::math::MathDialect>();
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<mlir::scf::SCFDialect>();
    registry.insert<mlir::ub::UBDialect>();
    registry.insert<numba::util::NumbaUtilDialect>();
    registry.insert<plier::PlierDialect>();
  }

  void runOnOperation() override;
};

struct BuildTupleConversionPattern
    : public mlir::OpConversionPattern<plier::BuildTupleOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::BuildTupleOp op, plier::BuildTupleOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    auto converter = getTypeConverter();
    assert(converter);
    auto retType = converter->convertType(op.getResult().getType());
    if (!retType.isa_and_nonnull<mlir::TupleType>())
      return mlir::failure();

    rewriter.replaceOpWithNewOp<numba::util::BuildTupleOp>(op, retType,
                                                           adaptor.getArgs());
    return mlir::success();
  }
};

template <typename Op, int Idx>
struct PairAccConversionPattern : public mlir::OpConversionPattern<Op> {
  using mlir::OpConversionPattern<Op>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    auto src = adaptor.getValue();
    if (!mlir::isa<mlir::TupleType>(src.getType()))
      return mlir::failure();

    auto converter = this->getTypeConverter();
    assert(converter);

    auto resType = converter->convertType(op.getType());
    if (!resType)
      return mlir::failure();

    auto loc = op.getLoc();
    auto idx = rewriter.create<mlir::arith::ConstantIndexOp>(loc, Idx);
    rewriter.replaceOpWithNewOp<numba::util::TupleExtractOp>(op, resType, src,
                                                             idx);
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
    assert(converter);

    auto resType = converter->convertType<mlir::TupleType>(op.getType());
    if (!resType || resType.size() != 5)
      return mlir::failure();

    auto src = adaptor.getValue();
    auto srcType = mlir::dyn_cast<mlir::TupleType>(src.getType());
    if (!srcType || srcType.size() != 3)
      return mlir::failure();

    auto loc = op.getLoc();
    auto getItem = [&](unsigned i) -> mlir::Value {
      auto idx = rewriter.create<mlir::arith::ConstantIndexOp>(loc, i);
      return rewriter.create<numba::util::TupleExtractOp>(
          loc, srcType.getType(i), src, idx);
    };

    auto begin = getItem(0);
    auto end = getItem(1);
    auto step = getItem(2);

    auto zero = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value isNeg = rewriter.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::slt, step, zero);
    auto getNeg = [&](mlir::Value val) -> mlir::Value {
      mlir::Value neg = rewriter.create<mlir::arith::SubIOp>(loc, zero, val);
      return rewriter.create<mlir::arith::SelectOp>(loc, isNeg, neg, val);
    };

    begin = getNeg(begin);
    end = getNeg(end);
    step = getNeg(step);

    auto iterType = mlir::MemRefType::get({}, rewriter.getIndexType());
    mlir::Value iter = rewriter.create<mlir::memref::AllocOp>(loc, iterType);
    rewriter.create<mlir::memref::StoreOp>(loc, begin, iter);

    mlir::Value rets[] = {begin, end, step, iter, isNeg};
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
    assert(converter);

    auto resType = converter->convertType<mlir::TupleType>(op.getType());
    if (!resType || resType.size() != 2)
      return mlir::failure();

    auto src = adaptor.getValue();
    auto srcType = mlir::dyn_cast<mlir::TupleType>(src.getType());
    if (!srcType || srcType.size() != 5)
      return mlir::failure();

    auto loc = op.getLoc();
    auto getItem = [&](unsigned i) -> mlir::Value {
      auto idx = rewriter.create<mlir::arith::ConstantIndexOp>(loc, i);
      return rewriter.create<numba::util::TupleExtractOp>(
          loc, srcType.getType(i), src, idx);
    };

    //    auto begin = getItem(0);
    auto end = getItem(1);
    auto step = getItem(2);
    auto iter = getItem(3);
    auto isNeg = getItem(4);

    mlir::Value current = rewriter.create<mlir::memref::LoadOp>(
        loc, iter, /*indices*/ std::nullopt);
    mlir::Value next = rewriter.create<mlir::arith::AddIOp>(loc, current, step);

    mlir::Value zero = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value currentNeg =
        rewriter.create<mlir::arith::SubIOp>(loc, zero, current);
    mlir::Value cond = rewriter.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::slt, current, end);

    rewriter.create<mlir::memref::StoreOp>(loc, next, iter,
                                           /*indices*/ std::nullopt);

    current =
        rewriter.create<mlir::arith::SelectOp>(loc, isNeg, currentNeg, current);
    current = numba::doConvert(rewriter, loc, current, resType.getType(0));
    if (!current)
      return mlir::failure();

    cond = numba::doConvert(rewriter, loc, cond, resType.getType(1));
    if (!cond)
      return mlir::failure();

    mlir::Value rets[] = {current, cond};
    mlir::Value ret =
        rewriter.create<numba::util::BuildTupleOp>(loc, resType, rets);
    rewriter.replaceOp(op, ret);
    return mlir::success();
  }
};

struct GetItemTupleConversionPattern
    : public mlir::OpConversionPattern<plier::GetItemOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(plier::GetItemOp op, plier::GetItemOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    auto container = adaptor.getValue();
    auto containerType = container.getType().dyn_cast<mlir::TupleType>();
    if (!containerType)
      return mlir::failure();

    auto &converter = *getTypeConverter();

    auto retType = converter.convertType(op.getType());
    if (!retType)
      return mlir::failure();

    auto index = numba::indexCast(rewriter, op.getLoc(), adaptor.getIndex());

    rewriter.replaceOpWithNewOp<numba::util::TupleExtractOp>(op, retType,
                                                             container, index);
    return mlir::success();
  }
};

void PlierToStdPass::runOnOperation() {
  mlir::TypeConverter typeConverter;
  // Convert unknown types to itself
  typeConverter.addConversion([](mlir::Type type) { return type; });

  auto context = &getContext();
  typeConverter.addConversion([](plier::FunctionType type) {
    return numba::util::OpaqueType::get(type.getContext());
  });
  typeConverter.addConversion(
      [](mlir::Type type, llvm::SmallVectorImpl<mlir::Type> &retTypes)
          -> std::optional<mlir::LogicalResult> {
        if (isOmittedType(type))
          return mlir::success();

        return std::nullopt;
      });

  auto indexType = mlir::IndexType::get(context);
  auto indexMemref = mlir::MemRefType::get({}, indexType);
  auto i1 = mlir::IntegerType::get(context, 1);
  auto rangeStateType =
      mlir::TupleType::get(context, {indexType, indexType, indexType});
  auto rangeIterType = mlir::TupleType::get(
      context, {indexType, indexType, indexType, indexMemref, i1});
  typeConverter.addConversion(
      [rangeStateType](plier::RangeStateType type) { return rangeStateType; });
  typeConverter.addConversion(
      [rangeIterType](plier::RangeIterType type) { return rangeIterType; });

  numba::populateTupleTypeConverter(typeConverter);

  auto materializeCast = [](mlir::OpBuilder &builder, mlir::Type type,
                            mlir::ValueRange inputs,
                            mlir::Location loc) -> std::optional<mlir::Value> {
    if (inputs.size() == 1)
      return builder
          .create<mlir::UnrealizedConversionCastOp>(loc, type, inputs.front())
          .getResult(0);

    return std::nullopt;
  };
  typeConverter.addArgumentMaterialization(materializeCast);
  typeConverter.addSourceMaterialization(materializeCast);
  typeConverter.addTargetMaterialization(materializeCast);

  mlir::RewritePatternSet patterns(context);
  mlir::ConversionTarget target(*context);

  auto isNum = [&](mlir::Type t) -> bool {
    if (!t)
      return false;

    auto res = typeConverter.convertType(t);
    return mlir::isa_and_nonnull<mlir::IntegerType, mlir::FloatType,
                                 mlir::IndexType, mlir::ComplexType>(res);
  };

  auto isTuple = [&](mlir::Type t) -> bool {
    if (!t)
      return false;

    auto res = typeConverter.convertType(t);
    return res && mlir::isa<mlir::TupleType>(res);
  };

  target.addDynamicallyLegalOp<plier::BinOp>([&](plier::BinOp op) {
    auto lhsType = op.getLhs().getType();
    auto rhsType = op.getRhs().getType();
    if (op.getOp() == "+" && isTuple(lhsType) && isTuple(rhsType))
      return false;

    return !isNum(lhsType) || !isNum(rhsType) || !isNum(op.getType());
  });
  target.addDynamicallyLegalOp<plier::InplaceBinOp>(
      [&](plier::InplaceBinOp op) {
        auto lhsType = op.getLhs().getType();
        auto rhsType = op.getRhs().getType();
        if (op.getOp() == "+" && isTuple(lhsType) && isTuple(rhsType))
          return false;

        return !isNum(lhsType) || !isNum(rhsType) || !isNum(op.getType());
      });
  target.addDynamicallyLegalOp<plier::UnaryOp>([&](plier::UnaryOp op) {
    return !isNum(op.getValue().getType()) && !isNum(op.getType());
  });
  target.addDynamicallyLegalOp<plier::CastOp>([&](plier::CastOp op) {
    auto inputType = op.getValue().getType();
    if (isOmittedType(inputType))
      return false;

    auto srcType = typeConverter.convertType(inputType);
    auto dstType = typeConverter.convertType(op.getType());
    if (srcType == dstType && inputType != op.getType())
      return false;

    return srcType == dstType || !isNum(srcType) || !isNum(dstType);
  });
  target.addDynamicallyLegalOp<plier::ConstOp, plier::GlobalOp>(
      [&](mlir::Operation *op) {
        auto type = typeConverter.convertType(op->getResult(0).getType());
        if (!type)
          return true;

        return !isSupportedConst(type);
      });

  target.addDynamicallyLegalOp<plier::GetItemOp>(
      [&](plier::GetItemOp op) -> std::optional<bool> {
        auto type = typeConverter.convertType(op.getValue().getType());
        if (type.isa_and_nonnull<mlir::TupleType>())
          return false;

        return std::nullopt;
      });
  target.addDynamicallyLegalOp<plier::GetiterOp>(
      [&](plier::GetiterOp op) -> std::optional<bool> {
        return !mlir::isa<plier::RangeStateType>(op.getValue().getType());
      });
  target.addDynamicallyLegalOp<plier::IternextOp>(
      [&](plier::IternextOp op) -> std::optional<bool> {
        return !mlir::isa<plier::RangeIterType>(op.getValue().getType());
      });
  target.addIllegalOp<plier::BuildTupleOp, plier::PairfirstOp,
                      plier::PairsecondOp, plier::UndefOp>();
  target.addLegalOp<numba::util::BuildTupleOp, numba::util::TupleExtractOp>();
  target.addLegalDialect<mlir::arith::ArithDialect>();
  target.addLegalDialect<mlir::complex::ComplexDialect>();

  patterns.insert<
      // clang-format off
      InplaceBinOpLowering,
      BinOpLowering,
      BinOpTupleLowering,
      UnaryOpLowering,
      LowerCasts,
      UndefOpLowering,
      ConstOpLowering,
      LiteralLowering<plier::CastOp>,
      LiteralLowering<plier::GlobalOp>,
      OmittedLowering,
      BuildTupleConversionPattern,
      GetitertConversionPattern,
      IternextConversionPattern,
      PairAccConversionPattern<plier::PairfirstOp, 0>,
      PairAccConversionPattern<plier::PairsecondOp, 1>,
      GetItemTupleConversionPattern
      // clang-format on
      >(typeConverter, context);

  numba::populateControlFlowTypeConversionRewritesAndTarget(typeConverter,
                                                            patterns, target);
  numba::populateTupleTypeConversionRewritesAndTarget(typeConverter, patterns,
                                                      target);

  if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                std::move(patterns))))
    signalPassFailure();
}

static void populatePlierToStdPipeline(mlir::OpPassManager &pm) {
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(std::make_unique<PlierToStdPass>());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(std::make_unique<BuiltinCallsLoweringPass>());
  pm.addPass(numba::createForceInlinePass());
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(numba::createPromoteWhilePass());
  pm.addPass(mlir::createCanonicalizerPass());
}
} // namespace

void registerPlierToStdPipeline(numba::PipelineRegistry &registry) {
  registry.registerPipeline([](auto sink) {
    auto stage = getHighLoweringStage();
    sink(plierToStdPipelineName(), {plierToScfPipelineName()}, {stage.end},
         {plierToScfPipelineName()}, &populatePlierToStdPipeline);
  });
}

llvm::StringRef plierToStdPipelineName() { return "plier_to_std"; }
