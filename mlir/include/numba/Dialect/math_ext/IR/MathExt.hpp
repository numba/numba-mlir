// SPDX-FileCopyrightText: 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Interfaces/VectorInterfaces.h>

//===----------------------------------------------------------------------===//
// Math Dialect
//===----------------------------------------------------------------------===//

#include "numba/Dialect/math_ext/IR/MathExtOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// Math Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "numba/Dialect/math_ext/IR/MathExtOps.h.inc"
