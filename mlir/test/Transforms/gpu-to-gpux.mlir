// RUN: numba-mlir-opt --gpu-to-gpux --split-input-file %s | FileCheck %s

func.func @alloc() {
  // CHECK-LABEL: func @alloc()

  // CHECK: %[[queue:.*]] = gpu_runtime.create_gpu_queue

  // CHECK: %[[m0:.*]] = gpu_runtime.alloc %[[queue]] () : memref<13xf32, 1>
  %m0 = gpu.alloc () : memref<13xf32, 1>
  // CHECK: gpu_runtime.dealloc %[[queue]] %[[m0]] : memref<13xf32, 1>
  gpu.dealloc %m0 : memref<13xf32, 1>

  %t0 = gpu.wait async
  // CHECK: %[[m1:.*]], %[[t1:.*]] = gpu_runtime.alloc %[[queue]] async [{{.*}}] () : memref<13xf32, 1>
  %m1, %t1 = gpu.alloc async [%t0] () : memref<13xf32, 1>
  // CHECK: gpu_runtime.dealloc %[[queue]] async [%[[t1]]] %[[m1]] : memref<13xf32, 1>
  %t2 = gpu.dealloc async [%t1] %m1 : memref<13xf32, 1>

  // CHECK: %[[m2:.*]] = gpu_runtime.alloc %[[queue]] host_shared () : memref<13xf32, 1>
  %m2 = gpu.alloc host_shared () : memref<13xf32, 1>
  // CHECK: gpu_runtime.dealloc %[[queue]] %[[m2]] : memref<13xf32, 1>
  gpu.dealloc %m2 : memref<13xf32, 1>

  // CHECK: gpu_runtime.destroy_gpu_queue %[[queue]]
  return
}

// -----

// CHECK-LABEL: func @region()
func.func @region() -> (memref<10xf32>, memref<11xf32>, memref<12xf32>){
// CHECK: %[[S1:.*]] = gpu_runtime.create_gpu_queue
// CHECK: %[[S2:.*]] = gpu_runtime.create_gpu_queue, "test1"
// CHECK: %[[S3:.*]] = gpu_runtime.create_gpu_queue, "test2"
// CHECK: %[[A1:.*]] = gpu_runtime.alloc %[[S1]] () : memref<10xf32>
  %0 = gpu.alloc () : memref<10xf32>
// CHECK: %[[R1:.*]] = numba_util.env_region #gpu_runtime.region_desc<device = "test1", usm_type = "device", spirv_major_version = 1, spirv_minor_version = 1, has_fp16 = true, has_fp64 = false> -> memref<11xf32> {
  %1 = numba_util.env_region #gpu_runtime.region_desc<device = "test1", usm_type = "device", spirv_major_version = 1, spirv_minor_version = 1, has_fp16 = true, has_fp64 = false> -> memref<11xf32> {
// CHECK: %[[A2:.*]] = gpu_runtime.alloc %[[S2]] () : memref<11xf32>
    %2 = gpu.alloc () : memref<11xf32>
// CHECK: numba_util.env_region_yield %[[A2]] : memref<11xf32>
    numba_util.env_region_yield %2: memref<11xf32>
  }
// CHECK: %[[R2:.*]] = numba_util.env_region #gpu_runtime.region_desc<device = "test2", usm_type = "device", spirv_major_version = 1, spirv_minor_version = 1, has_fp16 = true, has_fp64 = false> -> memref<12xf32> {
  %3 = numba_util.env_region #gpu_runtime.region_desc<device = "test2", usm_type = "device", spirv_major_version = 1, spirv_minor_version = 1, has_fp16 = true, has_fp64 = false> -> memref<12xf32> {
// CHECK: %[[A3:.*]] = gpu_runtime.alloc %[[S3]] () : memref<12xf32>
    %4 = gpu.alloc () : memref<12xf32>
// CHECK: numba_util.env_region_yield %[[A3]] : memref<12xf32>
    numba_util.env_region_yield %4: memref<12xf32>
  }

// CHECK: gpu_runtime.destroy_gpu_queue %[[S3]]
// CHECK: gpu_runtime.destroy_gpu_queue %[[S2]]
// CHECK: gpu_runtime.destroy_gpu_queue %[[S1]]

// CHECK: return %[[A1]], %[[R1]], %[[R2]]
  return %0, %1, %3 : memref<10xf32>, memref<11xf32>, memref<12xf32>
}
