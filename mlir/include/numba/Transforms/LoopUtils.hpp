// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>

namespace mlir {
class Location;
class OpBuilder;
class Operation;
class PatternRewriter;
class Region;
class Type;
class Value;
struct LogicalResult;
namespace scf {
class ForOp;
class WhileOp;
} // namespace scf
} // namespace mlir

namespace plier {
class GetiterOp;
}

namespace numba {
bool canLowerWhileToFor(mlir::scf::WhileOp whileOp);
llvm::SmallVector<mlir::scf::ForOp, 2> lowerWhileToFor(
    mlir::scf::WhileOp whileOp, mlir::PatternRewriter &builder,
    llvm::function_ref<std::tuple<mlir::Value, mlir::Value, mlir::Value>(
        mlir::OpBuilder &, mlir::Location)>
        getBounds,
    llvm::function_ref<mlir::Value(mlir::OpBuilder &, mlir::Location,
                                   mlir::Type, mlir::Value)>
        getIterVal);
mlir::LogicalResult lowerWhileToFor(
    plier::GetiterOp getiter, mlir::PatternRewriter &builder,
    llvm::function_ref<std::tuple<mlir::Value, mlir::Value, mlir::Value>(
        mlir::OpBuilder &, mlir::Location)>
        getBounds,
    llvm::function_ref<mlir::Value(mlir::OpBuilder &, mlir::Location,
                                   mlir::Type, mlir::Value)>
        getIterVal,
    llvm::function_ref<void(mlir::scf::ForOp)> results = nullptr);

mlir::LogicalResult naivelyFuseParallelOps(mlir::Region &region);
mlir::LogicalResult
prepareForFusion(mlir::Region &region,
                 llvm::function_ref<bool(mlir::Operation &)> needPrepare);
} // namespace numba
