// RUN: numba-mlir-opt %s -allow-unregistered-dialect -canonicalize --split-input-file | FileCheck %s

func.func @test(%arg1: index, %arg2: i64) -> i64 {
  %0 = numba_util.build_tuple %arg1, %arg2: index, i64 -> tuple<index, i64>
  %cst = arith.constant 1 : index
  %1 = numba_util.tuple_extract %0 : tuple<index, i64>, %cst -> i64
  return %1 : i64
}
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: index, %[[ARG2:.*]]: i64)
//  CHECK-NEXT:   return %[[ARG2]] : i64

// -----

func.func @remove_empty_region() {
  numba_util.env_region "test" {
  }
  return
}
// CHECK-LABEL: func @remove_empty_region
//   CHECK-NOT:   numba_util.env_region
//  CHECK-NEXT:   return

// -----

func.func @empty_region_out_value(%arg1: index) -> index {
  %0 = numba_util.env_region "test" -> index {
    numba_util.env_region_yield %arg1: index
  }
  return %0 : index
}
// CHECK-LABEL: func @empty_region_out_value
//  CHECK-SAME: (%[[ARG:.*]]: index)
//       CHECK: return %[[ARG]] : index

// -----

func.func @merge_nested_region() {
  numba_util.env_region "test" {
    "test.test1"() : () -> ()
    numba_util.env_region "test" {
      "test.test2"() : () -> ()
    }
    "test.test3"() : () -> ()
  }
  return
}
// CHECK-LABEL: func @merge_nested_region
//  CHECK-NEXT:   numba_util.env_region
//  CHECK-NEXT:   "test.test1"() : () -> ()
//  CHECK-NEXT:   "test.test2"() : () -> ()
//  CHECK-NEXT:   "test.test3"() : () -> ()
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return

// -----

func.func @nested_region_yield_args() {
  %0:4 = numba_util.env_region "test" -> index, i32, index, i64 {
    %1:3 = "test.test1"() : () -> (index, i32, i64)
    numba_util.env_region_yield %1#0, %1#1, %1#0, %1#2: index, i32, index, i64
  }
  "test.test2"(%0#0, %0#2, %0#3) : (index, index, i64) -> ()
  return
}
// CHECK-LABEL: func @nested_region_yield_args
//  CHECK-NEXT:   %[[RES:.*]]:2 = numba_util.env_region "test" -> index, i64 {
//  CHECK-NEXT:   %[[VAL:.*]]:3 = "test.test1"() : () -> (index, i32, i64)
//  CHECK-NEXT:   numba_util.env_region_yield %[[VAL]]#0, %[[VAL]]#2 : index, i64
//  CHECK-NEXT:   }
//  CHECK-NEXT:   "test.test2"(%[[RES]]#0, %[[RES]]#0, %[[RES]]#1) : (index, index, i64) -> ()
//  CHECK-NEXT:   return

// -----

func.func @merge_adjacent_region1() {
  numba_util.env_region "test" {
    "test.test1"() : () -> ()
  }
  numba_util.env_region "test" {
    "test.test2"() : () -> ()
  }
  numba_util.env_region "test" {
    "test.test3"() : () -> ()
  }
  return
}
// CHECK-LABEL: func @merge_adjacent_region1
//  CHECK-NEXT:   numba_util.env_region
//  CHECK-NEXT:   "test.test1"() : () -> ()
//  CHECK-NEXT:   "test.test2"() : () -> ()
//  CHECK-NEXT:   "test.test3"() : () -> ()
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return

// -----

func.func @merge_adjacent_region2() {
  %0 = numba_util.env_region "test" -> index {
    %1 = "test.test1"() : () -> index
    numba_util.env_region_yield %1: index
  }
  %2 = numba_util.env_region "test" -> i64 {
    %3 = "test.test2"(%0) : (index) -> i64
    numba_util.env_region_yield %3: i64
  }
  numba_util.env_region "test" {
    "test.test3"(%2) : (i64) -> ()
  }
  return
}
// CHECK-LABEL: func @merge_adjacent_region2
//  CHECK-NEXT:   numba_util.env_region
//  CHECK-NEXT:   %[[VAL1:.*]] = "test.test1"() : () -> index
//  CHECK-NEXT:   %[[VAL2:.*]] = "test.test2"(%[[VAL1]]) : (index) -> i64
//  CHECK-NEXT:   "test.test3"(%[[VAL2]]) : (i64) -> ()
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return

// -----

// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: tensor<?x?xf32>, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index)
//       CHECK:   return %[[ARG3]]
func.func @test(%arg1: tensor<?x?xf32>, %arg2: index, %arg3: index) -> index {
  %cst = arith.constant 1 : index
  %0 = numba_util.enforce_shape %arg1 : tensor<?x?xf32>(%arg2, %arg3) -> tensor<?x?xf32>
  %1 = tensor.dim %0, %cst : tensor<?x?xf32>
  return %1: index
}

// -----

// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: memref<?x?xf32>, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index)
//       CHECK:   return %[[ARG3]]
func.func @test(%arg1: memref<?x?xf32>, %arg2: index, %arg3: index) -> index {
  %cst = arith.constant 1 : index
  %0 = numba_util.enforce_shape %arg1 : memref<?x?xf32>(%arg2, %arg3) -> memref<?x?xf32>
  %1 = memref.dim %0, %cst : memref<?x?xf32>
  return %1: index
}

// -----

// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: !ntensor.ntensor<?x?xf32>, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index)
//       CHECK:   return %[[ARG3]]
func.func @test(%arg1: !ntensor.ntensor<?x?xf32>, %arg2: index, %arg3: index) -> index {
  %cst = arith.constant 1 : index
  %0 = numba_util.enforce_shape %arg1 : !ntensor.ntensor<?x?xf32>(%arg2, %arg3) -> !ntensor.ntensor<?x?xf32>
  %1 = ntensor.dim %0, %cst : !ntensor.ntensor<?x?xf32>
  return %1: index
}

// -----

// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: memref<?xf32>)
//  CHECK-NEXT:   %[[RES:.*]] = numba_util.retain %[[ARG]] : memref<?xf32> to memref<?xf32>
//  CHECK-NEXT:   return %[[RES]]
func.func @test(%arg1: memref<?xf32>) -> memref<?xf32> {
  %1 = numba_util.retain %arg1 : memref<?xf32> to memref<?xf32>
  memref.dealloc %1 : memref<?xf32>
  %2 = numba_util.retain %1 : memref<?xf32> to memref<?xf32>
  %3 = numba_util.retain %2 : memref<?xf32> to memref<?xf32>
  memref.dealloc %2 : memref<?xf32>
  return %3: memref<?xf32>
}

// -----

// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG:.*]]: memref<?xf32>)
//       CHECK:   %[[RES:.*]] = numba_util.get_alloc_token %[[ARG]] : memref<?xf32> -> index
//       CHECK:   return %[[RES]]
func.func @test(%arg: memref<?xf32>) -> index {
  %0 = memref.cast %arg : memref<?xf32> to memref<2xf32>
  %res = numba_util.get_alloc_token %0 : memref<2xf32> -> index
  return %res: index
}

// -----

// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG:.*]]: memref<?xf32>)
//       CHECK:   %[[RES:.*]] = numba_util.get_alloc_token %[[ARG]] : memref<?xf32> -> index
//       CHECK:   return %[[RES]]
func.func @test(%arg: memref<?xf32>) -> index {
  %0 = memref.reinterpret_cast %arg to offset: [0], sizes: [3], strides: [1] : memref<?xf32> to memref<3xf32>
  %res = numba_util.get_alloc_token %0 : memref<3xf32> -> index
  return %res: index
}

// -----

// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG:.*]]: memref<?xf32>)
//       CHECK:   %[[RES:.*]] = numba_util.get_alloc_token %[[ARG]] : memref<?xf32> -> index
//       CHECK:   return %[[RES]]
func.func @test(%arg: memref<?xf32>) -> index {
  %0, %offset, %sizes, %strides = memref.extract_strided_metadata %arg : memref<?xf32> -> memref<f32>, index, index, index
  %res = numba_util.get_alloc_token %0 : memref<f32> -> index
  return %res: index
}

// -----

// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG:.*]]: memref<?xf32>)
//       CHECK:   %[[RES:.*]] = numba_util.get_alloc_token %[[ARG]] : memref<?xf32> -> index
//       CHECK:   return %[[RES]]
func.func @test(%arg: memref<?xf32>) -> index {
  %0 = memref.subview %arg[1][3][1] : memref<?xf32> to memref<3xf32, strided<[1], offset: 1>>
  %res = numba_util.get_alloc_token %0 : memref<3xf32, strided<[1], offset: 1>> -> index
  return %res: index
}

// -----

// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: tensor<?xf32>, %[[ARG2]]: index)
//       CHECK:   %[[EXPAND:.*]] = tensor.expand_shape %[[ARG1]] {{\[}}[0, 1, 2]] output_shape [1, %dim, 1] : tensor<?xf32> into tensor<1x?x1xf32>
//       CHECK:   %[[RES:.*]] = tensor.cast %[[EXPAND]] : tensor<1x?x1xf32> to tensor<?x?x?xf32>
//       CHECK:   return %[[RES]]
func.func @test(%arg0: tensor<?xf32>, %arg1: index) -> tensor<?x?x?xf32> {
  %c1 = arith.constant 1 : index
  %0 = numba_util.reshape %arg0(%c1, %arg1, %c1) : (tensor<?xf32>, index, index, index) -> tensor<?x?x?xf32>
  return %0: tensor<?x?x?xf32>
}

// -----

// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: tensor<?xf32>, %[[ARG2]]: index)
//       CHECK:   %[[RES:.*]] = tensor.extract_slice %[[ARG1]][0] [%[[ARG2]]] [1] : tensor<?xf32> to tensor<?xf32>
//       CHECK:   return %[[RES]]
func.func @test(%arg0: tensor<?xf32>, %arg1: index) -> tensor<?xf32> {
  %c1 = arith.constant 1 : index
  %0 = numba_util.reshape %arg0(%arg1) : (tensor<?xf32>, index) -> tensor<?xf32>
  return %0: tensor<?xf32>
}

// -----

// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: !ntensor.ntensor<?xf32>, %[[ARG2]]: index)
//       CHECK:   %[[RES:.*]] = ntensor.subview %[[ARG1]][0] [%[[ARG2]]] [1] : !ntensor.ntensor<?xf32> to !ntensor.ntensor<?xf32>
//       CHECK:   return %[[RES]]
func.func @test(%arg0: !ntensor.ntensor<?xf32>, %arg1: index) -> !ntensor.ntensor<?xf32> {
  %c1 = arith.constant 1 : index
  %0 = numba_util.reshape %arg0(%arg1) : (!ntensor.ntensor<?xf32>, index) -> !ntensor.ntensor<?xf32>
  return %0: !ntensor.ntensor<?xf32>
}
