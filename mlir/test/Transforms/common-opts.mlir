// RUN: numba-mlir-opt -allow-unregistered-dialect --numba-common-opts --split-input-file %s | FileCheck %s

// CHECK-LABEL: func @test
//       CHECK:  %[[RES:.*]]:3 = scf.while (%[[ARG0:.*]] = %{{.*}}, %[[ARG1:.*]] = %{{.*}}, %[[ARG2:.*]] = %{{.*}}) : (f32, i32, i64) -> (f32, i32, i64) {
//       CHECK:  scf.condition(%{{.*}}) %[[ARG0]], %[[ARG1]], %[[ARG2]] : f32, i32, i64
//       CHECK:  ^bb0(%[[ARG3:.*]]: f32, %[[ARG4:.*]]: i32, %[[ARG5:.*]]: i64):
//       CHECK:  %[[R1:.*]] = "test.test"(%[[ARG5]]) : (i64) -> f32
//       CHECK:  %[[R2:.*]] = "test.test"(%[[ARG3]]) : (f32) -> i32
//       CHECK:  %[[R3:.*]] = "test.test"(%[[ARG4]]) : (i32) -> i64
//       CHECK:  scf.yield %[[R1]], %[[R2]], %[[R3]] : f32, i32, i64
//       CHECK:  return %[[RES]]#2, %[[RES]]#0, %[[RES]]#1
func.func @test() -> (i64, f32, i32) {
  %0 = "test.test"() : () -> (f32)
  %1 = "test.test"() : () -> (i32)
  %2 = "test.test"() : () -> (i64)
  %3:3 = scf.while (%arg0 = %0, %arg1 = %1, %arg2 = %2) : (f32, i32, i64) -> (i64, f32, i32) {
    %cond = "test.test"() : () -> (i1)
    scf.condition(%cond) %arg2, %arg0, %arg1 : i64, f32, i32
  } do {
  ^bb0(%arg3: i64, %arg4: f32, %arg5: i32):
    %4 = "test.test"(%arg3) : (i64) -> (f32)
    %5 = "test.test"(%arg4) : (f32) -> (i32)
    %6 = "test.test"(%arg5) : (i32) -> (i64)
    scf.yield %4, %5, %6 : f32, i32, i64
  }
  return %3#0, %3#1, %3#2 : i64, f32, i32
}
