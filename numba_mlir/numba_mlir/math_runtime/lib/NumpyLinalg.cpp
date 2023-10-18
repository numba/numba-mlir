// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cassert>
#include <stdio.h>
#include <string_view>

#include "Common.hpp"
#include "numba-mlir-math-runtime_export.h"

#ifdef NUMBA_MLIR_USE_DPNP
#include <dpnp_iface.hpp>
#endif

#ifdef NUMBA_MLIR_USE_MKL
#include "mkl.h"
#endif

namespace {

template <typename T>
void eigImpl(Memref<2, const T> *input, Memref<1, T> *vals,
             Memref<2, T> *vecs) {
#ifdef NUMBA_MLIR_USE_DPNP
  dpnp_eig_c<T, T>(input->data, vals->data, vecs->data, input->dims[0]);
#else
  (void)input;
  (void)vals;
  (void)vecs;
  // direct MKL call or another implementation?
  fprintf(stderr, "Math runtime was compiled without DPNP support\n");
  fflush(stderr);
  abort();
#endif
}

template <typename T>
static void checkSquare(const Memref<2, T> *arr, char arrName) {
  if (arr->dims[0] != arr->dims[1]) {
    fatal_failure("Array '%c' is not square,  dims are %d and %d.\n", arrName,
                  int(arr->dims[0]), int(arr->dims[1]));
  }
};

template <typename T>
static bool isEmpty2d(const Memref<2, T> *arr, char arrName) {
  if (arr->dims[0] < 0 || arr->dims[1] < 0) {
    fatal_failure("Array dims must not be negative. '%c' dims are %d and %d.\n",
                  arrName, int(arr->dims[0]), int(arr->dims[1]));
  }
  return arr->dims[0] == 0 && arr->dims[1] == 0;
};

template <typename T>
static void isContiguous2d(const Memref<2, T> *arr, char arrName) {
  if (arr->strides[0] <= 0 || arr->strides[1] <= 0) {
    fatal_failure("Strides must be positive. '%c' strides are %d and %d.\n",
                  arrName, int(arr->strides[0]), int(arr->strides[1]));
  }

  if (arr->strides[0] != 1 && arr->strides[1] != 1) {
    fatal_failure(
        "mkl gemm suports only arrays contiguous on inner dimension.\n"
        "stride for at least one dimension should be equal to 1.\n"
        "'%c' parameter is not contiguous. '%c' strides are %d and %d.\n",
        arrName, arrName, int(arr->strides[0]), int(arr->strides[1]));
  }
};

template <typename T> static bool isRowm(const Memref<2, T> *arr) {
  return arr->strides[1] == 1;
};

#ifdef NUMBA_MLIR_USE_MKL
template <typename T>
using GemmFunc = void(const CBLAS_LAYOUT, const CBLAS_TRANSPOSE,
                      const CBLAS_TRANSPOSE, const MKL_INT, const MKL_INT,
                      const MKL_INT, const T, const T *, const MKL_INT,
                      const T *, const MKL_INT, const T, T *, const MKL_INT);

template <typename T>
static void cpuGemm(GemmFunc<T> Gemm, const Memref<2, T> *a,
                    const Memref<2, T> *b, Memref<2, T> *c, T alpha, T beta) {
  assert(a);
  assert(b);
  assert(c);

  // Special case when we matmul empty arrays. Nothing to do in this case.
  if (isEmpty2d(a, 'a') && isEmpty2d(b, 'b'))
    return;

  isContiguous2d(a, 'a');
  isContiguous2d(b, 'b');
  isContiguous2d(c, 'c');

  auto layout = isRowm(c) ? CblasRowMajor : CblasColMajor;
  auto transA = isRowm(a) == isRowm(c) ? CblasNoTrans : CblasTrans;
  auto transB = isRowm(b) == isRowm(c) ? CblasNoTrans : CblasTrans;

  auto m = static_cast<MKL_INT>(a->dims[0]);
  auto n = static_cast<MKL_INT>(b->dims[1]);
  auto k = static_cast<MKL_INT>(a->dims[1]);

  auto lda = static_cast<MKL_INT>(isRowm(a) ? a->strides[0] : a->strides[1]);
  auto ldb = static_cast<MKL_INT>(isRowm(b) ? b->strides[0] : b->strides[1]);
  auto ldc = static_cast<MKL_INT>(isRowm(c) ? c->strides[0] : c->strides[1]);

  auto aData = getMemrefData(a);
  auto bData = getMemrefData(b);
  auto cData = getMemrefData(c);

  Gemm(layout, /*layout*/
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
       ldc     /*ldc*/
  );
}

template <typename T>
using GetrfFunc = lapack_int(int, lapack_int, lapack_int, T *, lapack_int,
                             lapack_int *);
template <typename T>
using GetriFunc = lapack_int(int, lapack_int, T *, lapack_int,
                             const lapack_int *);

template <typename T>
static int cpuInv(GetrfFunc<T> getrf, GetriFunc<T> getri, Memref<2, T> *a,
                  Memref<1, int64_t> *ipiv) {
  assert(a);
  assert(ipiv);

  // Nothing to do for empty arrays.
  if (isEmpty2d(a, 'a'))
    return 0;

  checkSquare(a, 'a');

  auto n = static_cast<MKL_INT>(a->dims[0]);

  auto layout = isRowm(a) ? CblasRowMajor : CblasColMajor;
  auto lda = static_cast<MKL_INT>(isRowm(a) ? a->strides[0] : a->strides[1]);

  if (auto res = getrf(layout, n, n, a->data, lda, ipiv->data))
    return static_cast<int>(res);

  return static_cast<int>((getri(layout, n, a->data, lda, ipiv->data)));
}

#endif
} // namespace

extern "C" {

#define EIG_VARIANT(T, Suff)                                                   \
  NUMBA_MLIR_MATH_RUNTIME_EXPORT void dpnp_linalg_eig_##Suff(                  \
      Memref<2, const T> *input, Memref<1, T> *vals, Memref<2, T> *vecs) {     \
    eigImpl(input, vals, vecs);                                                \
  }

EIG_VARIANT(float, float32)
EIG_VARIANT(double, float64)

#undef EIG_VARIANT

#ifdef NUMBA_MLIR_USE_MKL
#define MKL_CALL(f, ...) f(__VA_ARGS__)

#define MKL_GEMM(Prefix) cblas_##Prefix##gemm

#define MKL_GETRF(Prefix) LAPACKE_##Prefix##getrf
#define MKL_GETRI(Prefix) LAPACKE_##Prefix##getri
#else
static inline void ALL_UNUSED(int dummy, ...) { (void)dummy; }
#define MKL_CALL(f, ...)                                                       \
  ALL_UNUSED(0, __VA_ARGS__);                                                  \
  fatal_failure("Math runtime was compiled without MKL support\n");

#define MKL_GEMM(Prefix) 0

#define MKL_GETRF(Prefix) 0
#define MKL_GETRI(Prefix) 0
#endif

#define GEMM_VARIANT(T, Prefix, Suff)                                          \
  NUMBA_MLIR_MATH_RUNTIME_EXPORT void mkl_gemm_##Suff(                         \
      const Memref<2, T> *a, const Memref<2, T> *b, T alpha, T beta,           \
      Memref<2, T> *c) {                                                       \
    MKL_CALL(cpuGemm<T>, MKL_GEMM(Prefix), a, b, c, alpha, beta);              \
  }

GEMM_VARIANT(float, s, float32)
GEMM_VARIANT(double, d, float64)

#undef GEMM_VARIANT

#define INV_VARIANT(T, Prefix, Suff)                                           \
  NUMBA_MLIR_MATH_RUNTIME_EXPORT void mkl_inv_##Suff(                          \
      Memref<2, T> *a, Memref<1, int64_t> *ipiv) {                             \
    MKL_CALL(cpuInv<T>, MKL_GETRF(Prefix), MKL_GETRI(Prefix), a, ipiv);        \
  }

INV_VARIANT(float, s, float32)
INV_VARIANT(double, d, float64)

#undef INV_VARIANT
}
