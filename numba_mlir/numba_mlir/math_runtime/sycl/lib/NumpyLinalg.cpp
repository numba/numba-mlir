// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>
#include <mutex>
#include <string_view>
#include <unordered_map>

#include "Common.hpp"
#include "numba-mlir-math-sycl-runtime_export.h"

#ifdef NUMBA_MLIR_USE_SYCL_MKL
#include "CL/sycl.hpp"
#include "oneapi/mkl.hpp"
#endif

#include "GpuCommon.hpp"

namespace {

#ifdef NUMBA_MLIR_USE_SYCL_MKL

static auto getDeviceSelector(std::string deviceName) {
  using Sel = sycl::ext::oneapi::filter_selector;
  return [selector = Sel(std::move(deviceName))](
             const sycl::device &dev) -> int { return selector(dev); };
}

struct QueueMap {
  std::unordered_map<std::string, cl::sycl::queue> map;
  std::mutex m;
  cl::sycl::queue getQueue(std::string device) {
    std::lock_guard<std::mutex> guard(m);
    auto device_queue_iter = map.find(device);
    if (device_queue_iter == map.end()) {
      try {
        sycl::device d{getDeviceSelector(device)};
        device_queue_iter = map.insert({device, sycl::queue(d)}).first;
      } catch (const sycl::exception &e) {
        std::vector<cl::sycl::device> devices = cl::sycl::device::get_devices();
        std::string known_devices = "";
        for (auto &&d : devices)
          known_devices += d.get_info<cl::sycl::info::device::name>() + ",";
        fatal_failure("Failed to find device matching name '%s'.\n"
                      "Error message is '%s'\n."
                      "Known devices ara: '%s'",
                      device.c_str(), e.what(), known_devices.c_str());
      }
    }

    return device_queue_iter->second;
  }
};

static std::unique_ptr<QueueMap> qMapPtr;

static cl::sycl::queue getQueue(numba::GPUQueueInterface *queueIface) {
  assert(queueIface);
  if (auto queue = queueIface->getQueue())
    return *queue;

  assert(qMapPtr);
  return qMapPtr->getQueue(std::string(queueIface->getDeviceName()));
}

template <typename T>
static void deviceGemm(void *queueObj, const Memref<2, T> *a,
                       const Memref<2, T> *b, Memref<2, T> *c, T alpha,
                       T beta) {
  auto queueIface = static_cast<numba::GPUQueueInterface *>(queueObj);

  auto isContiguous = [](const Memref<2, T> *arr, char arr_name) {
    if (arr->strides[0] != 1 && arr->strides[1] != 1) {
      fatal_failure(
          "mkl gemm suports only arrays contiguous on inner dimension.\n"
          "stride for at least one dimension should be equal to 1.\n"
          "'%c' parameter is not contiguous. '%c' strides are %d and %d.\n",
          arr_name, arr_name, int(arr->strides[0]), int(arr->strides[1]));
    }
  };

  isContiguous(a, 'a');
  isContiguous(b, 'b');
  isContiguous(c, 'c');

  auto isRowm = [](const Memref<2, T> *arr) { return arr->strides[1] == 1; };
  auto transA = isRowm(a) == isRowm(c) ? oneapi::mkl::transpose::N
                                       : oneapi::mkl::transpose::T;
  auto transB = isRowm(b) == isRowm(c) ? oneapi::mkl::transpose::N
                                       : oneapi::mkl::transpose::T;

  auto m = static_cast<std::int64_t>(a->dims[0]);
  auto n = static_cast<std::int64_t>(b->dims[1]);
  auto k = static_cast<std::int64_t>(a->dims[1]);

  auto lda =
      static_cast<std::int64_t>(isRowm(a) ? a->strides[0] : a->strides[1]);
  auto ldb =
      static_cast<std::int64_t>(isRowm(b) ? b->strides[0] : b->strides[1]);
  auto ldc =
      static_cast<std::int64_t>(isRowm(c) ? c->strides[0] : c->strides[1]);

  auto aData = getMemrefData(a);
  auto bData = getMemrefData(b);
  auto cData = getMemrefData(c);

  auto queue = getQueue(queueIface);

  if (isRowm(c)) {
    oneapi::mkl::blas::row_major::gemm(queue,  /*queue*/
                                       transA, /*transa*/
                                       transB, /*transb*/
                                       m,      /*m*/
                                       n,      /*n*/
                                       k,      /*k*/
                                       alpha,  /*alpha*/
                                       aData,  /*a*/
                                       lda,    /*lda*/
                                       bData,  /*b*/
                                       ldb,    /*ldb*/
                                       beta,   /*beta*/
                                       cData,  /*c*/
                                       ldc,    /*ldc*/
                                       {}      /*dependencies*/
                                       )
        .wait();
  } else {
    oneapi::mkl::blas::column_major::gemm(queue,  /*queue*/
                                          transA, /*transa*/
                                          transB, /*transb*/
                                          m,      /*m*/
                                          n,      /*n*/
                                          k,      /*k*/
                                          alpha,  /*alpha*/
                                          aData,  /*a*/
                                          lda,    /*lda*/
                                          bData,  /*b*/
                                          ldb,    /*ldb*/
                                          beta,   /*beta*/
                                          cData,  /*c*/
                                          ldc,    /*ldc*/
                                          {}      /*dependencies*/
                                          )
        .wait();
  }
}
#endif

void initMap() {
#ifdef NUMBA_MLIR_USE_SYCL_MKL
  qMapPtr.reset(new QueueMap());
#endif
}

void finilizeMap() {
#ifdef NUMBA_MLIR_USE_SYCL_MKL
  qMapPtr.reset();
#endif
}

} // namespace

extern "C" {
#ifdef NUMBA_MLIR_USE_SYCL_MKL
#define GEMM_VARIANT(T, Suff)                                                  \
  NUMBA_MLIR_MATH_SYCL_RUNTIME_EXPORT void mkl_gemm_##Suff##_device(           \
      void *queue, const Memref<2, T> *a, const Memref<2, T> *b, T alpha,      \
      T beta, Memref<2, T> *c) {                                               \
    deviceGemm<T>(queue, a, b, c, alpha, beta);                                \
  }

GEMM_VARIANT(float, float32)
GEMM_VARIANT(double, float64)
#undef GEMM_VARIANT
#endif

// Not thread safe
NUMBA_MLIR_MATH_SYCL_RUNTIME_EXPORT void nmrtMathRuntimeInit() { initMap(); }

NUMBA_MLIR_MATH_SYCL_RUNTIME_EXPORT void nmrtMathRuntimeFinalize() {
  finilizeMap();
}
}
