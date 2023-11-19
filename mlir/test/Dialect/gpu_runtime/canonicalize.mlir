// RUN: numba-mlir-opt %s -allow-unregistered-dialect -canonicalize --split-input-file | FileCheck %s

// CHECK-LABEL: @test
//       CHECK: numba_util.env_region #gpu_runtime.region_desc
//   CHECK-NOT: numba_util.env_region #gpu_runtime.region_desc
//       CHECK: "test.test"() : () -> ()
func.func @test() {
  numba_util.env_region #gpu_runtime.region_desc<device = "test", usm_type = "device", spirv_major_version = 1, spirv_minor_version = 1, has_fp16 = true, has_fp64 = false> {
    numba_util.env_region "test" {
      numba_util.env_region #gpu_runtime.region_desc<device = "test", usm_type = "device", spirv_major_version = 1, spirv_minor_version = 1, has_fp16 = true, has_fp64 = false> {
        "test.test"() : () -> ()
      }
    }
  }
  return
}
