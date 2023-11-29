// RUN: numba-mlir-opt -pass-pipeline='builtin.module(func.func(copy-removal))' -allow-unregistered-dialect --split-input-file %s | FileCheck %s

// CHECK-LABEL: func @test
//  CHECK-SAME: (%[[ARG1:.*]]: !ntensor.ntensor<?xf32>, %[[ARG2:.*]]: !ntensor.ntensor<?xf32>)
//  CHECK-NEXT: ntensor.copy %[[ARG1]], %[[ARG2]] : !ntensor.ntensor<?xf32> to !ntensor.ntensor<?xf32>
//  CHECK-NEXT: return
func.func @test(%arg1: !ntensor.ntensor<?xf32>, %arg2: !ntensor.ntensor<?xf32>) {
  ntensor.copy %arg1, %arg2 : !ntensor.ntensor<?xf32> to !ntensor.ntensor<?xf32>
  ntensor.copy %arg1, %arg2 : !ntensor.ntensor<?xf32> to !ntensor.ntensor<?xf32>
  return
}

// -----

// CHECK-LABEL: func @test
//  CHECK-SAME: (%[[ARG:.*]]: !ntensor.ntensor<?xf32>)
//       CHECK: %[[RES1:.*]] = ntensor.primitive "foo" () -> !ntensor.ntensor<?xf32>
//       CHECK: ntensor.copy %[[RES1]], %[[ARG]] : !ntensor.ntensor<?xf32> to !ntensor.ntensor<?xf32>
//       CHECK: %[[RES2:.*]] = ntensor.primitive "bar" (%[[RES1]]) : !ntensor.ntensor<?xf32> -> !ntensor.ntensor<?xf32>
//       CHECK: return %[[RES2]]
func.func @test(%arg1: !ntensor.ntensor<?xf32>) -> !ntensor.ntensor<?xf32> {
  %0 = ntensor.primitive "foo" () -> !ntensor.ntensor<?xf32>
  ntensor.copy %0, %arg1 : !ntensor.ntensor<?xf32> to !ntensor.ntensor<?xf32>
  %1 = ntensor.primitive "bar" (%arg1) : !ntensor.ntensor<?xf32> -> !ntensor.ntensor<?xf32>
  return %1 : !ntensor.ntensor<?xf32>
}

// -----

// CHECK: #[[MAP:.*]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func @test
//  CHECK-SAME: (%[[ARG0:.*]]: memref<?xi64>, %[[ARG1:.*]]: memref<?xi64>)
//  CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
//  CHECK-NEXT: %[[DIM:.*]] = memref.dim %[[ARG0]], %[[C0]] : memref<?xi64>
//  CHECK-NEXT: linalg.generic {indexing_maps = [#[[MAP]], #[[MAP]]], iterator_types = ["parallel"]} ins(%[[ARG0]] : memref<?xi64>) outs(%[[ARG1]] : memref<?xi64>) {
//  CHECK-NEXT: ^bb0(%[[IN:.*]]: i64, %{{.*}}: i64):
//  CHECK-NEXT:   linalg.yield %[[IN]] : i64
//  CHECK-NEXT: }
//  CHECK-NEXT: return
#map = affine_map<(d0) -> (d0)>
func.func @test(%arg0: memref<?xi64>, %arg1: memref<?xi64>) {
  %c0 = arith.constant 0 : index
  %dim = memref.dim %arg0, %c0 : memref<?xi64>
  %alloc = memref.alloc(%dim) : memref<?xi64>
  linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg0 : memref<?xi64>) outs(%alloc : memref<?xi64>) {
  ^bb0(%in: i64, %out: i64):
    linalg.yield %in : i64
  }
  linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%alloc : memref<?xi64>) outs(%arg1 : memref<?xi64>) {
  ^bb0(%in: i64, %out: i64):
    linalg.yield %in : i64
  }
  return
}

// -----

// In this case pass shouldn't do any changes
// CHECK: #[[MAP:.*]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func @test
//  CHECK-SAME: (%[[ARG0:.*]]: memref<?xi64, strided<[-1], offset: ?>>, %[[ARG1:.*]]: memref<?xi64>)
//  CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
//  CHECK-NEXT: %[[DIM:.*]] = memref.dim %[[ARG0]], %[[C0]] : memref<?xi64, strided<[-1], offset: ?>>
//  CHECK-NEXT: %[[ALLOC:.*]] = memref.alloc(%[[DIM]]) : memref<?xi64>
//  CHECK-NEXT: linalg.generic {indexing_maps = [#[[MAP]], #[[MAP]]], iterator_types = ["parallel"]} ins(%[[ARG0]] : memref<?xi64, strided<[-1], offset: ?>>) outs(%[[ALLOC]] : memref<?xi64>) {
//  CHECK-NEXT: ^bb0(%[[IN:.*]]: i64, %{{.*}}: i64):
//  CHECK-NEXT:   linalg.yield %[[IN]] : i64
//  CHECK-NEXT: }
//  CHECK-NEXT: linalg.generic {indexing_maps = [#[[MAP]], #[[MAP]]], iterator_types = ["parallel"]} ins(%[[ALLOC]] : memref<?xi64>) outs(%[[ARG1]] : memref<?xi64>) {
//  CHECK-NEXT: ^bb0(%[[IN:.*]]: i64, %{{.*}}: i64):
//  CHECK-NEXT:   linalg.yield %[[IN]] : i64
//  CHECK-NEXT: }
//  CHECK-NEXT: return
#map = affine_map<(d0) -> (d0)>
func.func @test(%arg0: memref<?xi64, strided<[-1], offset: ?>>, %arg1: memref<?xi64>) {
  %c0 = arith.constant 0 : index
  %dim = memref.dim %arg0, %c0 : memref<?xi64, strided<[-1], offset: ?>>
  %alloc = memref.alloc(%dim) : memref<?xi64>
  linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg0 : memref<?xi64, strided<[-1], offset: ?>>) outs(%alloc : memref<?xi64>) {
  ^bb0(%in: i64, %out: i64):
    linalg.yield %in : i64
  }
  linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%alloc : memref<?xi64>) outs(%arg1 : memref<?xi64>) {
  ^bb0(%in: i64, %out: i64):
    linalg.yield %in : i64
  }
  return
}

// -----

// CHECK-LABEL: func @test
//  CHECK-SAME: (%[[ARG1:.*]]: memref<?xf32>, %[[ARG2:.*]]: memref<?xf32>)
//       CHECK: scf.for
//       CHECK: %{{.*}} = memref.load %[[ARG1]][%{{.*}}] : memref<?xf32>
//       CHECK: }
//       CHECK: return
#map = affine_map<(d0) -> (d0)>
func.func @test(%arg1: memref<?xf32>, %arg2: memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %alloc = memref.alloc(%c10) : memref<?xf32>
  linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg2 : memref<?xf32>) outs(%arg1 : memref<?xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  }
  scf.for %arg5 = %c0 to %c10 step %c1 {
    %alloc2 = memref.alloc(%c10) : memref<?xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg1 : memref<?xf32>) outs(%alloc2 : memref<?xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
    }
    %1 = memref.load %alloc2[%c0] : memref<?xf32>
    "test.test"(%1) : (f32) -> ()
  }
  return
}

// -----

// CHECK-LABEL: func @test
//  CHECK-SAME: (%[[ARG1:.*]]: memref<?xf32>, %[[ARG2:.*]]: memref<?xf32>)
//       CHECK: scf.for
//       CHECK: %{{.*}} = memref.load %[[ARG1]][%{{.*}}] : memref<?xf32>
//       CHECK: memref.store %{{.*}}, %[[ARG1]][%{{.*}}] : memref<?xf32>
//       CHECK: }
//       CHECK: return
#map = affine_map<(d0) -> (d0)>
func.func @test(%arg1: memref<?xf32>, %arg2: memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %alloc = memref.alloc(%c10) : memref<?xf32>
  linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg2 : memref<?xf32>) outs(%arg1 : memref<?xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  }
  scf.for %arg5 = %c0 to %c10 step %c1 {
    %alloc2 = memref.alloc(%c10) : memref<?xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg1 : memref<?xf32>) outs(%alloc2 : memref<?xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
    }
    %1 = memref.load %alloc2[%c0] : memref<?xf32>
    "test.test"(%1) : (f32) -> ()
    %2 = "test.test"() : () -> (f32)
    memref.store %2, %arg1[%c0] : memref<?xf32>
  }
  return
}

