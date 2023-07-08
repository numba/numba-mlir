// RUN: numba-mlir-opt -allow-unregistered-dialect --numba-sort-loops-for-gpu --split-input-file %s | FileCheck %s

// CHECK-LABEL: check
//  CHECK-SAME: (%[[ARR1:.*]]: memref<?x?xf64>, %[[ARR2:.*]]: memref<?x?xf64>, %[[DIM1:.*]]: index, %[[DIM2:.*]]: index)
//       CHECK: %[[O:.*]] = arith.constant 1 : index
//       CHECK: %[[Z:.*]] = arith.constant 0 : index
//       CHECK: scf.parallel (%[[IDX1:.*]], %[[IDX2:.*]]) = (%[[Z]], %[[Z]]) to (%[[DIM2]], %[[DIM1]]) step (%[[O]], %[[O]]) {
//       CHECK: %[[VAL:.*]] = memref.load %[[ARR1]][%[[IDX2]], %[[IDX1]]] : memref<?x?xf64>
//       CHECK: memref.store %[[VAL]], %[[ARR2]][%[[IDX2]], %[[IDX1]]] : memref<?x?xf64>
func.func @check(%arg0: memref<?x?xf64>, %arg1: memref<?x?xf64>, %arg2: index, %arg3: index) {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  numba_util.env_region #gpu_runtime.region_desc<device = "test"> {
    scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%arg2, %arg3) step (%c1, %c1) {
      %2 = memref.load %arg0[%arg4, %arg5] : memref<?x?xf64>
      memref.store %2, %arg1[%arg4, %arg5] : memref<?x?xf64>
      scf.yield
    }
  }
  return
}

// -----

// CHECK-LABEL: check
//  CHECK-SAME: (%[[ARR1:.*]]: memref<?x?x?xf64>, %[[ARR2:.*]]: memref<?x?x?xf64>,
//  CHECK-SAME: %[[DIM1:.*]]: index, %[[DIM2:.*]]: index, %[[DIM3:.*]]: index)
//       CHECK: %[[O:.*]] = arith.constant 1 : index
//       CHECK: %[[Z:.*]] = arith.constant 0 : index
//       CHECK: scf.parallel (%[[IDX1:.*]], %[[IDX2:.*]], %[[IDX3:.*]]) = (%[[Z]], %[[Z]], %[[Z]])
//  CHECK-SAME: to (%[[DIM3]], %[[DIM2]], %[[DIM1]]) step (%[[O]], %[[O]], %[[O]]) {
//       CHECK: %[[VAL:.*]] = memref.load %[[ARR1]][%[[IDX3]], %[[IDX2]], %[[IDX1]]] : memref<?x?x?xf64>
//       CHECK: memref.store %[[VAL]], %[[ARR2]][%[[IDX3]], %[[IDX2]], %[[IDX1]]] : memref<?x?x?xf64>
func.func @check(%arg0: memref<?x?x?xf64>, %arg1: memref<?x?x?xf64>, %arg3: index, %arg4: index, %arg5: index) {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  numba_util.env_region #gpu_runtime.region_desc<device = "test"> {
    scf.parallel (%arg6, %arg7, %arg8) = (%c0, %c0, %c0) to (%arg3, %arg4, %arg5) step (%c1, %c1, %c1) {
      %2 = memref.load %arg0[%arg6, %arg7, %arg8] : memref<?x?x?xf64>
      memref.store %2, %arg1[%arg6, %arg7, %arg8] : memref<?x?x?xf64>
      scf.yield
    }
  }
  return
}

// -----

// CHECK-LABEL: check
//  CHECK-SAME: (%[[ARR1:.*]]: memref<?x?x?xf64>, %[[ARR2:.*]]: memref<?x?x?xf64>,
//  CHECK-SAME: %[[DIM1:.*]]: index, %[[DIM2:.*]]: index, %[[DIM3:.*]]: index)
//       CHECK: %[[O:.*]] = arith.constant 1 : index
//       CHECK: %[[Z:.*]] = arith.constant 0 : index
//       CHECK: scf.parallel (%[[IDX1:.*]], %[[IDX2:.*]], %[[IDX3:.*]]) = (%[[Z]], %[[Z]], %[[Z]])
//  CHECK-SAME: to (%[[DIM2]], %[[DIM3]], %[[DIM1]]) step (%[[O]], %[[O]], %[[O]]) {
//       CHECK: %[[VAL:.*]] = memref.load %[[ARR1]][%[[IDX3]], %[[IDX2]], %[[IDX1]]] : memref<?x?x?xf64>
//       CHECK: memref.store %[[VAL]], %[[ARR2]][%[[IDX3]], %[[IDX2]], %[[IDX1]]] : memref<?x?x?xf64>
func.func @check(%arg0: memref<?x?x?xf64>, %arg1: memref<?x?x?xf64>, %arg3: index, %arg4: index, %arg5: index) {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  numba_util.env_region #gpu_runtime.region_desc<device = "test"> {
    scf.parallel (%arg6, %arg7, %arg8) = (%c0, %c0, %c0) to (%arg3, %arg4, %arg5) step (%c1, %c1, %c1) {
      %2 = memref.load %arg0[%arg6, %arg8, %arg7] : memref<?x?x?xf64>
      memref.store %2, %arg1[%arg6, %arg8, %arg7] : memref<?x?x?xf64>
      scf.yield
    }
  }
  return
}
