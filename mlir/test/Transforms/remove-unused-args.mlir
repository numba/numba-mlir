// RUN: numba-mlir-opt -allow-unregistered-dialect --numba-remove-unused-args --split-input-file %s | FileCheck %s


// CHECK-LABEL: func private @test(%{{.*}}: index)
func.func private @test(%arg1: index) {
  "test.test"(%arg1) : (index) -> ()
  return
}

// -----

// CHECK-LABEL: func private @test()
func.func private @test(%arg1: index) {
  return
}

// -----

// CHECK-LABEL: func private @test()
func.func private @test(%arg1: index {test.test}) {
  return
}

// -----

// CHECK-LABEL: func private @test(index)
func.func private @test(index)

// -----

// CHECK-LABEL: func @test(%{{.*}}: index)
func.func @test(%arg1: index) {
  return
}

// -----

// CHECK-LABEL: func private @test()
func.func private @test(%arg1: index) {
  func.call @test(%arg1) : (index) -> ()
  "test.test"() : () -> ()
  return
}

// -----

// CHECK-LABEL: func @test(%{{.*}}: index)
// CHECK: call @test1() : () -> ()
func.func @test(%arg1: index) {
  func.call @test1(%arg1) : (index) -> ()
  return
}

// CHECK-LABEL: func private @test1()
func.func private @test1(%arg1: index) {
  return
}

// -----

// CHECK-LABEL: func @test(%{{.*}}: index)
// CHECK: call @test1() : () -> ()
func.func @test(%arg1: index) {
  func.call @test1(%arg1) : (index) -> ()
  return
}

// CHECK-LABEL: func private @test1()
// CHECK: call @test2() : () -> ()
func.func private @test1(%arg1: index) {
  func.call @test2(%arg1) : (index) -> ()
  return
}

// CHECK-LABEL: func private @test2()
func.func private @test2(%arg1: index) {
  return
}
