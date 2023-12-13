// SPDX-FileCopyrightText: 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef MATH_EXT_BASE
#define MATH_EXT_BASE
include "mlir/IR/OpBase.td"
def MathExt_Dialect : Dialect {
  let name = "math_ext";
  let cppNamespace = "::numba::math_ext";
  let description = [{
    Extensions to math dialect

    The math dialect is intended to hold mathematical operations on integer and
    floating types beyond simple arithmetics. Each operation works on scalar, vector
    or tensor type. On vector and tensor type operations apply elementwise unless
    explicitly specified otherwise. As an example, the floating point absolute value
    can be expressed as:

    ```mlir
    // Scalar absolute value.
    %a = math.absf %b : f64

    // Vector elementwise absolute value.
    %f = math.absf %g : vector<4xf32>

    // Tensor elementwise absolute value.
    %x = math.absf %y : tensor<4x?xf8>
    ```
  }];
  let hasConstantMaterializer = 1;
  let dependentDialects = [
    "::mlir::arith::ArithDialect"
  ];
}
#endif // MATH_EXT_BASE
