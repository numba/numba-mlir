// RUN: numba-mlir-opt --numba-shape-int-range-opts --allow-unregistered-dialect --split-input-file %s | FileCheck %s

// CHECK-LABEL: func @test
//       CHECK:   %[[C:.*]] = arith.constant false
//       CHECK:   return %[[C]]
func.func @test(%arg1: tensor<?xf32>) -> i1 {
  %cst = arith.constant 0 : index
  %cst1 = arith.constant -1 : index
  %0 = tensor.dim %arg1, %cst : tensor<?xf32>
  %1 = arith.cmpi eq, %0, %cst1 : index
  return %1: i1
}

// -----

// CHECK-LABEL: func @test
//       CHECK:   %[[C:.*]] = arith.constant true
//       CHECK:   return %[[C]]
func.func @test(%arg1: tensor<?xf32>) -> i1 {
  %cst = arith.constant 0 : index
  %cst1 = arith.constant -1 : index
  %0 = tensor.dim %arg1, %cst : tensor<?xf32>
  %1 = arith.cmpi ne, %0, %cst1 : index
  return %1: i1
}

// -----


// CHECK-LABEL: func @test
//       CHECK:   %[[C:.*]] = arith.constant true
//       CHECK:   return %[[C]]
func.func @test(%arg1: tensor<?xf32>) -> i1 {
  %cst = arith.constant 0 : index
  %0 = tensor.dim %arg1, %cst : tensor<?xf32>
  %1 = arith.cmpi sge, %0, %cst : index
  return %1: i1
}

// -----

// CHECK-LABEL: func @test
//       CHECK:   %[[C:.*]] = arith.constant false
//       CHECK:   return %[[C]]
func.func @test(%arg1: tensor<?xf32>) -> i1 {
  %cst = arith.constant 0 : index
  %0 = tensor.dim %arg1, %cst : tensor<?xf32>
  %1 = arith.cmpi slt, %0, %cst : index
  return %1: i1
}

// -----


// CHECK-LABEL: func @test
//       CHECK:   %[[C:.*]] = arith.constant true
//       CHECK:   return %[[C]]
func.func @test(%arg1: tensor<?xf32>) -> i1 {
  %cst = arith.constant 0 : index
  %cst1 = arith.constant -1 : index
  %0 = tensor.dim %arg1, %cst : tensor<?xf32>
  %1 = arith.cmpi sgt, %0, %cst1 : index
  return %1: i1
}

// -----

// CHECK-LABEL: func @test
//       CHECK:   %[[C:.*]] = arith.constant false
//       CHECK:   return %[[C]]
func.func @test(%arg1: tensor<?xf32>) -> i1 {
  %cst = arith.constant 0 : index
  %cst1 = arith.constant -1 : index
  %0 = tensor.dim %arg1, %cst : tensor<?xf32>
  %1 = arith.cmpi sle, %0, %cst1 : index
  return %1: i1
}

// -----

// CHECK-LABEL: func @test
//       CHECK:   %[[C:.*]] = arith.constant false
//       CHECK:   return %[[C]]
func.func @test(%arg1: tensor<?xf32> {numba.shape_range = [#numba_util.index_range<[1,10]>]}) -> i1 {
  %cst = arith.constant 0 : index
  %0 = tensor.dim %arg1, %cst : tensor<?xf32>
  %1 = arith.cmpi eq, %0, %cst : index
  return %1: i1
}

// -----

// CHECK-LABEL: func @test
//       CHECK:   %[[C:.*]] = arith.constant false
//       CHECK:   return %[[C]]
func.func @test(%arg1: tensor<?xf32> {numba.shape_range = [#numba_util.index_range<[1,7]>]},
                %arg2: tensor<?xf32> {numba.shape_range = [#numba_util.index_range<[2,10]>]},
                %cond: i1) -> i1 {
  %cst = arith.constant 0 : index
  %0 = arith.select %cond, %arg1, %arg2 : tensor<?xf32>
  %1 = tensor.dim %0, %cst : tensor<?xf32>
  %2 = arith.cmpi eq, %1, %cst : index
  return %2: i1
}

// -----

// CHECK-LABEL: func @test
//       CHECK:   %[[C:.*]] = arith.constant false
//       CHECK:   return %[[C]]
func.func @test(%arg1: tensor<?xf32>) -> i1 {
  %cst0 = arith.constant 0 : index
  %cst1 = arith.constant 1 : index
  %1 = numba_util.reshape %arg1(%cst0, %cst1) : (tensor<?xf32>, index, index) -> tensor<?x?xf32>
  %2 = tensor.dim %1, %cst0 : tensor<?x?xf32>
  %3 = arith.cmpi eq, %2, %cst1 : index
  return %3: i1
}

// -----

// CHECK-LABEL: func @test
//       CHECK:   %[[C:.*]] = arith.constant false
//       CHECK:   return %[[C]]
func.func @test(%arg1: tensor<?xf32>) -> i1 {
  %cst0 = arith.constant 0 : index
  %cst1 = arith.constant 1 : index
  %1 = numba_util.reshape %arg1(%cst0, %cst1) : (tensor<?xf32>, index, index) -> tensor<?x?xf32>
  %2 = tensor.dim %1, %cst1 : tensor<?x?xf32>
  %3 = arith.cmpi eq, %2, %cst0 : index
  return %3: i1
}

// -----

// CHECK-LABEL: func @test
//       CHECK:   %[[C:.*]] = arith.constant false
//       CHECK:   return %[[C]]
func.func @test(%arg1: tensor<?xf32>, %arg2: tensor<?xf32> {numba.shape_range = [#numba_util.index_range<[2,10]>]}) -> i1 {
  %cst0 = arith.constant 0 : index
  %cst1 = arith.constant 1 : index
  %0 = tensor.dim %arg2, %cst0 : tensor<?xf32>
  %1 = tensor.extract_slice %arg1[0] [%0] [1] : tensor<?xf32> to tensor<?xf32>
  %2 = tensor.dim %1, %cst0 : tensor<?xf32>
  %3 = arith.cmpi eq, %2, %cst1 : index
  return %3: i1
}

// -----

// CHECK-LABEL: func @test
//       CHECK:   %[[C:.*]] = arith.constant false
//       CHECK:   return %[[C]]
func.func @test(%arg1: tensor<?xf32>, %arg2: tensor<?xf32> {numba.shape_range = [#numba_util.index_range<[2,10]>]}) -> i1 {
  %cst0 = arith.constant 0 : index
  %cst1 = arith.constant 1 : index
  %0 = tensor.dim %arg2, %cst0 : tensor<?xf32>
  %1 = numba_util.enforce_shape %arg1 : tensor<?xf32>(%0) -> tensor<?xf32>
  %2 = tensor.dim %1, %cst0 : tensor<?xf32>
  %3 = arith.cmpi eq, %2, %cst1 : index
  return %3: i1
}

// -----

// CHECK-LABEL: func @test
//       CHECK:   %[[C:.*]] = arith.constant false
//       CHECK:   return %[[C]]
func.func @test(%arg1: tensor<?xf32> {numba.shape_range = [#numba_util.index_range<[2,10]>]}) -> i1 {
  %cst0 = arith.constant 0 : index
  %cst1 = arith.constant 1 : index
  %0 = tensor.dim %arg1, %cst0 : tensor<?xf32>
  %1 = tensor.empty(%0) : tensor<?xf32>
  %2 = tensor.dim %1, %cst0 : tensor<?xf32>
  %3 = arith.cmpi eq, %2, %cst1 : index
  return %3: i1
}

// -----

// CHECK-LABEL: func private @test_nested(
//       CHECK:   %[[C:.*]] = arith.constant false
//       CHECK:   return %[[C]]
// CHECK-LABEL: func @test(
func.func private @test_nested(%arg1: tensor<?xf32>) -> i1 {
  %cst = arith.constant 0 : index
  %0 = tensor.dim %arg1, %cst : tensor<?xf32>
  %1 = arith.cmpi eq, %0, %cst : index
  return %1: i1
}

func.func @test(%arg1: tensor<?xf32> {numba.shape_range = [#numba_util.index_range<[2,10]>]}) -> i1 {
  %0 = func.call @test_nested(%arg1) : (tensor<?xf32>) -> i1
  return %0: i1
}

// -----

// CHECK-LABEL: func @test
//       CHECK:   %[[C:.*]] = arith.constant false
//       CHECK:   "test.test"(%[[C]]) : (i1) -> ()
func.func @test(%arg1: tensor<?xf32> {numba.shape_range = [#numba_util.index_range<[1,10]>]}, %arg2: i1) {
  %cst = arith.constant 0 : index
  scf.if %arg2 {
    %0 = tensor.dim %arg1, %cst : tensor<?xf32>
    %1 = arith.cmpi eq, %0, %cst : index
    "test.test"(%1) : (i1) -> ()
  }
  return
}

// -----

// CHECK-LABEL: func private @test_nested(
//       CHECK:   %[[C:.*]] = arith.constant false
//       CHECK:   "test.test"(%[[C]]) : (i1) -> ()
// CHECK-LABEL: func @test(
func.func private @test_nested(%arg1: tensor<?xf32>, %arg2: i1) {
  %cst = arith.constant 0 : index
  scf.if %arg2 {
    %0 = tensor.dim %arg1, %cst : tensor<?xf32>
    %1 = arith.cmpi eq, %0, %cst : index
    "test.test"(%1) : (i1) -> ()
  }
  return
}

func.func @test(%arg1: tensor<?xf32> {numba.shape_range = [#numba_util.index_range<[2,10]>]}, %arg2: i1) {
  func.call @test_nested(%arg1, %arg2) : (tensor<?xf32>, i1) -> ()
  return
}

// -----

// CHECK-LABEL: func @test
//       CHECK:   %[[C:.*]] = arith.constant false
//       CHECK:   "test.test"(%[[C]]) : (i1) -> ()
func.func @test(%arg1: tensor<?xf32> {numba.shape_range = [#numba_util.index_range<[1,10]>]}, %arg2: f32) {
  %cst = arith.constant 0 : index
  %cst1 = arith.constant 0 : i32
  %1 = llvm.fptosi %arg2 : f32 to i32
  %2 = arith.cmpi eq, %1, %cst1 : i32
  scf.if %2 {
    %3 = tensor.dim %arg1, %cst : tensor<?xf32>
    %4 = arith.cmpi eq, %3, %cst : index
    "test.test"(%4) : (i1) -> ()
  }
  return
}

// -----

// CHECK-LABEL: func @test
//       CHECK:   %[[C:.*]] = arith.constant false
//       CHECK:   return %[[C]]

#map0 = affine_map<(d0) -> (d0)>
func.func @test(%arg1: tensor<?xf32>, %arg2: tensor<?xf32> {numba.shape_range = [#numba_util.index_range<[2,10]>]}) -> i1 {
  %cst0 = arith.constant 0 : index
  %cst1 = arith.constant 1 : index
  %1 = linalg.generic {
      indexing_maps = [#map0, #map0],
      iterator_types = ["parallel"]}
      ins(%arg1 : tensor<?xf32>)
      outs(%arg2 : tensor<?xf32>) {
        ^bb0(%arg3: f32, %arg4: f32):
          linalg.yield %arg3 : f32
      } -> tensor<?xf32>
  %2 = tensor.dim %1, %cst0 : tensor<?xf32>
  %3 = arith.cmpi eq, %2, %cst1 : index
  return %3: i1
}

// -----

// CHECK-LABEL: func @test
//       CHECK:   %[[C:.*]] = arith.constant false
//       CHECK:   return %[[C]]
func.func @test(%arg1: tensor<?xf32> {numba.shape_range = [#numba_util.index_range<[1,10]>]}) -> i1 {
  %cst0 = arith.constant 0 : index
  %cst1 = arith.constant 1 : index
  %0 = numba_util.env_region "test" -> index {
    %1 = tensor.dim %arg1, %cst0 : tensor<?xf32>
    %2 = tensor.empty(%1) : tensor<?xf32>
    %3 = tensor.dim %2, %cst0 : tensor<?xf32>
    "test.test"(%2) : (tensor<?xf32>) -> ()
    numba_util.env_region_yield %3: index
  }
  %3 = arith.cmpi eq, %0, %cst0 : index
  return %3: i1
}

// -----

// CHECK-LABEL: func @test
//       CHECK1:   %[[C:.*]] = arith.constant false
//       CHECK:   return %[[C]]
func.func @test(%arg1: tensor<?xf32> {numba.shape_range = [#numba_util.index_range<[1,10]>]}) -> i1 {
  %cst0 = arith.constant 0 : index
  %cst1 = arith.constant 1 : index
  %0 = numba_util.env_region "test" -> tensor<?xf32> {
    %1 = tensor.dim %arg1, %cst0 : tensor<?xf32>
    %2 = tensor.empty(%1) : tensor<?xf32>
    "test.test"(%2) : (tensor<?xf32>) -> ()
    numba_util.env_region_yield %2: tensor<?xf32>
  }
  %2 = tensor.dim %0, %cst0 : tensor<?xf32>
  %3 = arith.cmpi eq, %2, %cst0 : index
  return %3: i1
}
