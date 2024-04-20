// RUN: numba-mlir-opt -allow-unregistered-dialect --gpux-make-barriers-uniform --split-input-file %s | FileCheck %s

func.func @test() {
  %c1 = arith.constant 1 : index
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)
             threads(%tx, %ty, %tz) in (%block_x = %c1, %block_y = %c1, %block_z = %c1) {
    %cond = "test.test1"() : () -> i1
    scf.if %cond {
      "test.test2"() : () -> ()
      %1 = "test.test3"() : () -> i32
      gpu.barrier {test.test_attr}
      "test.test4"() : () -> ()
      "test.test5"(%1) : (i32) -> ()
    }
    gpu.terminator
  }
  return
}

// CHECK-LABEL: func @test
//       CHECK: %[[V2:.*]] = ub.poison : i32
//       CHECK: gpu.launch blocks
//       CHECK: %[[COND:.*]] = "test.test1"() : () -> i1
//       CHECK: %[[RES1:.*]] = scf.if %[[COND]] -> (i32) {
//       CHECK: "test.test2"() : () -> ()
//       CHECK: %[[V1:.*]] = "test.test3"() : () -> i32
//       CHECK: scf.yield %[[V1]] : i32
//       CHECK: } else {
//       CHECK: scf.yield %[[V2]] : i32
//       CHECK: }
//       CHECK: gpu.barrier {test.test_attr}
//       CHECK: scf.if %[[COND]] {
//       CHECK: "test.test4"() : () -> ()
//       CHECK: "test.test5"(%[[RES1]]) : (i32) -> ()
//       CHECK: }
//       CHECK: gpu.terminator
//       CHECK: return

// -----

func.func @test() {
  %c1 = arith.constant 1 : index
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)
             threads(%tx, %ty, %tz) in (%block_x = %c1, %block_y = %c1, %block_z = %c1) {
    %cond = "test.test1"() : () -> i1
    scf.if %cond {
      "test.test2"() : () -> ()
      %1 = "test.test3"() : () -> i32
      %2 = "test.test4"() : () -> i64
      gpu.barrier
      "test.test5"() : () -> ()
      "test.test6"(%1) : (i32) -> ()
      %3 = "test.test7"() : () -> index
      gpu.barrier {test.test_attr}
      "test.test8"() : () -> ()
      "test.test9"(%2) : (i64) -> ()
      "test.test10"(%3) : (index) -> ()
    }
    gpu.terminator
  }
  return
}

// CHECK-LABEL: func @test
//   CHECK-DAG: %[[V6:.*]] = ub.poison : index
//   CHECK-DAG: %[[V4:.*]] = ub.poison : i64
//   CHECK-DAG: %[[V3:.*]] = ub.poison : i32
//       CHECK: gpu.launch blocks
//       CHECK: %[[COND:.*]] = "test.test1"() : () -> i1
//       CHECK: %[[RES1:.*]]:2 = scf.if %[[COND]] -> (i32, i64) {
//       CHECK: "test.test2"() : () -> ()
//       CHECK: %[[V1:.*]] = "test.test3"() : () -> i32
//       CHECK: %[[V2:.*]] = "test.test4"() : () -> i64
//       CHECK: scf.yield %[[V1]], %[[V2]] : i32, i64
//       CHECK: } else {
//       CHECK: scf.yield %[[V3]], %[[V4]] : i32, i64
//       CHECK: }
//       CHECK: gpu.barrier
//       CHECK: %[[RES2:.*]] = scf.if %[[COND]] -> (index) {
//       CHECK: "test.test5"() : () -> ()
//       CHECK: "test.test6"(%[[RES1]]#0) : (i32) -> ()
//       CHECK: %[[V5:.*]] = "test.test7"() : () -> index
//       CHECK: scf.yield %[[V5]] : index
//       CHECK: } else {
//       CHECK: scf.yield %[[V6]] : index
//       CHECK: }
//       CHECK: gpu.barrier {test.test_attr}
//       CHECK: scf.if %[[COND]] {
//       CHECK: "test.test8"() : () -> ()
//       CHECK: "test.test9"(%[[RES1]]#1) : (i64) -> ()
//       CHECK: "test.test10"(%[[RES2]]) : (index) -> ()
//       CHECK: }
//       CHECK: gpu.terminator
//       CHECK: return

// -----

func.func @test() {
  %c1 = arith.constant 1 : index
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)
             threads(%tx, %ty, %tz) in (%block_x = %c1, %block_y = %c1, %block_z = %c1) {
    %cond = "test.test1"() : () -> i1
    scf.if %cond {
      "test.test2"() : () -> ()
      %1 = "test.test3"() : () -> i32
      %2 = "test.test4"() : () -> i64
      %3 = gpu.all_reduce %2 uniform {
      ^bb(%lhs : i64, %rhs : i64):
        %xor = arith.xori %lhs, %rhs : i64
        gpu.yield %xor : i64
      } : (i64) -> (i64)
      "test.test5"() : () -> ()
      "test.test6"(%1) : (i32) -> ()
      "test.test7"(%3) : (i64) -> ()
    }
    gpu.terminator
  }
  return
}

// CHECK-LABEL: func @test
//   CHECK-DAG: %[[V4:.*]] = ub.poison : i64
//   CHECK-DAG: %[[V3:.*]] = ub.poison : i32
//   CHECK-DAG: %[[NEUTRAL:.*]] = arith.constant 0 : i64
//       CHECK: gpu.launch blocks
//       CHECK: %[[COND:.*]] = "test.test1"() : () -> i1
//       CHECK: %[[RES1:.*]]:2 = scf.if %[[COND]] -> (i32, i64) {
//       CHECK: "test.test2"() : () -> ()
//       CHECK: %[[V1:.*]] = "test.test3"() : () -> i32
//       CHECK: %[[V2:.*]] = "test.test4"() : () -> i64
//       CHECK: scf.yield %[[V1]], %[[V2]] : i32, i64
//       CHECK: } else {
//       CHECK: scf.yield %[[V3]], %[[V4]] : i32, i64
//       CHECK: }
//       CHECK: %[[RARG:.*]] = arith.select %[[COND]], %[[RES1]]#1, %[[NEUTRAL]] : i64
//       CHECK: %[[RRES:.*]] = gpu.all_reduce %[[RARG]] uniform {
//       CHECK:  ^bb0(%[[A1:.*]]: i64, %[[A2:.*]]: i64):
//       CHECK:   %[[X:.*]] = arith.xori %[[A1]], %[[A2]] : i64
//       CHECK:   gpu.yield %[[X]] : i64
//       CHECK:   } : (i64) -> i64
//       CHECK: scf.if %[[COND]] {
//       CHECK: "test.test5"() : () -> ()
//       CHECK: "test.test6"(%[[RES1]]#0) : (i32) -> ()
//       CHECK: "test.test7"(%[[RRES]]) : (i64) -> ()
//       CHECK: }
//       CHECK: gpu.terminator
//       CHECK: return

// -----

func.func @test() {
  %c1 = arith.constant 1 : index
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)
             threads(%tx, %ty, %tz) in (%block_x = %c1, %block_y = %c1, %block_z = %c1) {
    %cond = "test.test1"() : () -> i1
    scf.if %cond {
      "test.test2"() : () -> ()
      %1 = "test.test3"() : () -> i32
      %2 = "test.test4"() : () -> i64
      %3 = gpu.all_reduce mul %2 {} : (i64) -> (i64)
      "test.test5"() : () -> ()
      "test.test6"(%1) : (i32) -> ()
      "test.test7"(%3) : (i64) -> ()
    }
    gpu.terminator
  }
  return
}

// CHECK-LABEL: func @test
//   CHECK-DAG: %[[V4:.*]] = ub.poison : i64
//   CHECK-DAG: %[[V3:.*]] = ub.poison : i32
//   CHECK-DAG: %[[NEUTRAL:.*]] = arith.constant 1 : i64
//       CHECK: gpu.launch blocks
//       CHECK: %[[COND:.*]] = "test.test1"() : () -> i1
//       CHECK: %[[RES1:.*]]:2 = scf.if %[[COND]] -> (i32, i64) {
//       CHECK: "test.test2"() : () -> ()
//       CHECK: %[[V1:.*]] = "test.test3"() : () -> i32
//       CHECK: %[[V2:.*]] = "test.test4"() : () -> i64
//       CHECK: scf.yield %[[V1]], %[[V2]] : i32, i64
//       CHECK: } else {
//       CHECK: scf.yield %[[V3]], %[[V4]] : i32, i64
//       CHECK: }
//       CHECK: %[[RARG:.*]] = arith.select %[[COND]], %[[RES1]]#1, %[[NEUTRAL]] : i64
//       CHECK: %[[RRES:.*]] = gpu.all_reduce mul %[[RARG]] uniform {
//       CHECK:   } : (i64) -> i64
//       CHECK: scf.if %[[COND]] {
//       CHECK: "test.test5"() : () -> ()
//       CHECK: "test.test6"(%[[RES1]]#0) : (i32) -> ()
//       CHECK: "test.test7"(%[[RRES]]) : (i64) -> ()
//       CHECK: }
//       CHECK: gpu.terminator
//       CHECK: return

// -----

func.func @test() {
  %c1 = arith.constant 1 : index
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)
             threads(%tx, %ty, %tz) in (%block_x = %c1, %block_y = %c1, %block_z = %c1) {
    %cond = "test.test1"() : () -> i1
    scf.if %cond {
      "test.test2"() : () -> ()
      %1 = "test.test3"() : () -> i32
      %2 = "test.test4"() : () -> i64
      %3 = gpu.subgroup_reduce minsi %2 {} : (i64) -> (i64)
      "test.test5"() : () -> ()
      "test.test6"(%1) : (i32) -> ()
      "test.test7"(%3) : (i64) -> ()
    }
    gpu.terminator
  }
  return
}

// CHECK-LABEL: func @test
//   CHECK-DAG: %[[V4:.*]] = ub.poison : i64
//   CHECK-DAG: %[[V3:.*]] = ub.poison : i32
//   CHECK-DAG: %[[NEUTRAL:.*]] = arith.constant 9223372036854775807 : i64
//       CHECK: gpu.launch blocks
//       CHECK: %[[COND:.*]] = "test.test1"() : () -> i1
//       CHECK: %[[RES1:.*]]:2 = scf.if %[[COND]] -> (i32, i64) {
//       CHECK: "test.test2"() : () -> ()
//       CHECK: %[[V1:.*]] = "test.test3"() : () -> i32
//       CHECK: %[[V2:.*]] = "test.test4"() : () -> i64
//       CHECK: scf.yield %[[V1]], %[[V2]] : i32, i64
//       CHECK: } else {
//       CHECK: scf.yield %[[V3]], %[[V4]] : i32, i64
//       CHECK: }
//       CHECK: %[[RARG:.*]] = arith.select %[[COND]], %[[RES1]]#1, %[[NEUTRAL]] : i64
//       CHECK: %[[RRES:.*]] = gpu.subgroup_reduce minsi %[[RARG]] uniform : (i64) -> i64
//       CHECK: scf.if %[[COND]] {
//       CHECK: "test.test5"() : () -> ()
//       CHECK: "test.test6"(%[[RES1]]#0) : (i32) -> ()
//       CHECK: "test.test7"(%[[RRES]]) : (i64) -> ()
//       CHECK: }
//       CHECK: gpu.terminator
//       CHECK: return

// -----

func.func @test() {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)
             threads(%tx, %ty, %tz) in (%block_x = %c1, %block_y = %c1, %block_z = %c1) {
    %cond = "test.test1"() : () -> i1
    scf.if %cond {
      "test.test2"() : () -> ()
      %1 = "test.test3"() : () -> i32
      gpu.barrier
      scf.for %i0 = %c1 to %c10 step %c1 {
        "test.test4"() : () -> ()
        "test.test5"(%1) : (i32) -> ()
      }
    }
    gpu.terminator
  }
  return
}

// CHECK-LABEL: func @test
//       CHECK: %[[V2:.*]] = ub.poison : i32
//       CHECK: gpu.launch blocks
//       CHECK: %[[COND:.*]] = "test.test1"() : () -> i1
//       CHECK: %[[RES1:.*]] = scf.if %[[COND]] -> (i32) {
//       CHECK: "test.test2"() : () -> ()
//       CHECK: %[[V1:.*]] = "test.test3"() : () -> i32
//       CHECK: scf.yield %[[V1]] : i32
//       CHECK: } else {
//       CHECK: scf.yield %[[V2]] : i32
//       CHECK: }
//       CHECK: gpu.barrier
//       CHECK: scf.if %[[COND]] {
//       CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       CHECK: "test.test4"() : () -> ()
//       CHECK: "test.test5"(%[[RES1]]) : (i32) -> ()
//       CHECK: }
//       CHECK: }
//       CHECK: gpu.terminator
//       CHECK: return
