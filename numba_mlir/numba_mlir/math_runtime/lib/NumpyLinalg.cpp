// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cassert>
#include <stdio.h>
#include <string_view>

#include "Common.hpp"
#include "numba-mlir-math-runtime_export.h"

#ifdef NUMBA_MLIR_USE_MKL
#include "mkl.h"
#endif

namespace {
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
static void gemmImpl(GemmFunc<T> Gemm, const Memref<2, T> *a,
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
static int invImpl(GetrfFunc<T> getrf, GetriFunc<T> getri, Memref<2, T> *a,
                   Memref<1, MKL_INT> *ipiv) {
  assert(a);
  assert(ipiv);

  // Nothing to do for empty arrays.
  if (isEmpty2d(a, 'a'))
    return 0;

  checkSquare(a, 'a');

  auto n = static_cast<MKL_INT>(a->dims[0]);

  auto layout = isRowm(a) ? CblasRowMajor : CblasColMajor;
  auto data = a->data;
  auto ipivData = ipiv->data;

  auto lda = static_cast<MKL_INT>(isRowm(a) ? a->strides[0] : a->strides[1]);

  if (auto res = getrf(layout, n, n, data, lda, ipivData))
    return static_cast<int>(res);

  return static_cast<int>(getri(layout, n, data, lda, ipivData));
}

template <typename T>
using GesvFunc = lapack_int(int, lapack_int, lapack_int, T *, lapack_int,
                            lapack_int *, T *, lapack_int);

template <typename T>
static int solveImpl(GesvFunc<T> gesv, Memref<2, T> *a, Memref<2, T> *b,
                     Memref<1, MKL_INT> *ipiv) {
  assert(a);
  assert(b);
  assert(ipiv);

  // Nothing to do for empty arrays.
  if (isEmpty2d(a, 'a'))
    return 0;

  checkSquare(a, 'a');

  auto n = static_cast<MKL_INT>(a->dims[0]);
  auto nrhs = static_cast<MKL_INT>(b->dims[1]);

  auto layout = isRowm(a) ? CblasRowMajor : CblasColMajor;

  auto aData = a->data;
  auto bData = b->data;
  auto ipivData = ipiv->data;

  auto lda = static_cast<MKL_INT>(isRowm(a) ? a->strides[0] : a->strides[1]);
  auto ldb = static_cast<MKL_INT>(isRowm(b) ? b->strides[0] : b->strides[1]);

  return static_cast<int>(
      gesv(layout, n, nrhs, aData, lda, ipivData, bData, ldb));
}

template <typename T>
using PotrfFunc = lapack_int(int, char, lapack_int, T *, lapack_int);

template <typename T>
static int choleskyImpl(PotrfFunc<T> potrf, Memref<2, T> *a) {
  assert(a);

  // Nothing to do for empty arrays.
  if (isEmpty2d(a, 'a'))
    return 0;

  checkSquare(a, 'a');

  auto n = static_cast<MKL_INT>(a->dims[0]);

  auto layout = CblasColMajor;
  auto data = a->data;
  auto lda = static_cast<MKL_INT>(a->strides[0]);

  return static_cast<int>(potrf(layout, 'U', n, data, lda));
}

template <typename T>
using GeevRealFunc = lapack_int(int, char, char, lapack_int, T *, lapack_int,
                                T *, T *, T *, lapack_int, T *, lapack_int);

template <typename T>
static int eigImplReal(GeevRealFunc<T> geev, char jobvl, char jobvr,
                       Memref<2, T> *a, Memref<1, T> *wr, Memref<1, T> *wi,
                       Memref<2, T> *vl, Memref<2, T> *vr) {
  assert(a);
  assert(wr);
  assert(wi);
  assert(vl);
  assert(vr);

  // Nothing to do for empty arrays.
  if (isEmpty2d(a, 'a'))
    return 0;

  checkSquare(a, 'a');

  auto n = static_cast<MKL_INT>(a->dims[0]);
  auto layout = CblasColMajor;
  auto data = a->data;
  auto lda = static_cast<MKL_INT>(a->strides[0]);

  auto wrData = wr->data;
  auto wiData = wi->data;

  auto vlData = vl->data;
  auto ldvl = static_cast<MKL_INT>(vl->strides[0]);

  auto vrData = vr->data;
  auto ldvr = static_cast<MKL_INT>(vr->strides[0]);

  return static_cast<int>(geev(layout, jobvl, jobvr, n, data, lda, wrData,
                               wiData, vlData, ldvl, vrData, ldvr));
}

template <typename T>
using GeevComplexFunc = lapack_int(int, char, char, lapack_int, T *, lapack_int,
                                   T *, T *, lapack_int, T *, lapack_int);

template <typename T>
static int eigImplComplex(GeevComplexFunc<T> geev, char jobvl, char jobvr,
                          Memref<2, T> *a, Memref<1, T> *w, Memref<2, T> *vl,
                          Memref<2, T> *vr) {
  assert(a);
  assert(w);
  assert(vl);
  assert(vr);

  // Nothing to do for empty arrays.
  if (isEmpty2d(a, 'a'))
    return 0;

  checkSquare(a, 'a');

  auto n = static_cast<MKL_INT>(a->dims[0]);
  auto layout = CblasColMajor;
  auto data = a->data;
  auto lda = static_cast<MKL_INT>(a->strides[0]);

  auto wData = w->data;

  auto vlData = vl->data;
  auto ldvl = static_cast<MKL_INT>(vl->strides[0]);

  auto vrData = vr->data;
  auto ldvr = static_cast<MKL_INT>(vr->strides[0]);

  return static_cast<int>(geev(layout, jobvl, jobvr, n, data, lda, wData,
                               vlData, ldvl, vrData, ldvr));
}

template <typename T, typename TW>
using XxxevdFunc = lapack_int(int, char, char, lapack_int, T *, lapack_int,
                              TW *);

template <typename T, typename TW>
static int eighImpl(XxxevdFunc<T, TW> xxxevd, char jobz, char uplo,
                    Memref<2, T> *a, Memref<1, TW> *w) {
  assert(a);
  assert(w);

  // Nothing to do for empty arrays.
  if (isEmpty2d(a, 'a'))
    return 0;

  checkSquare(a, 'a');

  auto n = static_cast<MKL_INT>(a->dims[0]);
  auto layout = CblasRowMajor;
  auto data = a->data;
  auto lda = static_cast<MKL_INT>(a->strides[0]);

  auto wData = w->data;

  return static_cast<int>(xxxevd(layout, jobz, uplo, n, data, lda, wData));
}

#endif
} // namespace

extern "C" {
#ifdef NUMBA_MLIR_USE_MKL
#define MKL_CALL(f, ...) f(__VA_ARGS__)

#define MKL_GEMM(Prefix) cblas_##Prefix##gemm

#define MKL_GETRF(Prefix) LAPACKE_##Prefix##getrf
#define MKL_GETRI(Prefix) LAPACKE_##Prefix##getri
#define MKL_GETSV(Prefix) LAPACKE_##Prefix##gesv
#define MKL_POTRF(Prefix) LAPACKE_##Prefix##potrf
#define MKL_GEEV(Prefix) LAPACKE_##Prefix##geev
#define MKL_SYEVD(Prefix) LAPACKE_##Prefix##syevd
#define MKL_HEEVD(Prefix) LAPACKE_##Prefix##heevd
#else
static inline void ALL_UNUSED(int dummy, ...) { (void)dummy; }
#define MKL_CALL(f, ...)                                                       \
  ALL_UNUSED(0, __VA_ARGS__);                                                  \
  fatal_failure("Math runtime was compiled without MKL support\n");

#define MKL_GEMM(Prefix) 0

#define MKL_GETRF(Prefix) 0
#define MKL_GETRI(Prefix) 0
#define MKL_GETSV(Prefix) 0
#define MKL_POTRF(Prefix) 0
#define MKL_GEEV(Prefix) 0
#define MKL_SYEVD(Prefix) 0
#define MKL_HEEVD(Prefix) 0
#endif

#define GEMM_VARIANT(T, Prefix, Suff)                                          \
  NUMBA_MLIR_MATH_RUNTIME_EXPORT void mkl_gemm_##Suff(                         \
      const Memref<2, T> *a, const Memref<2, T> *b, T alpha, T beta,           \
      Memref<2, T> *c) {                                                       \
    MKL_CALL(gemmImpl<T>, MKL_GEMM(Prefix), a, b, c, alpha, beta);             \
  }

GEMM_VARIANT(float, s, float32)
GEMM_VARIANT(double, d, float64)

#undef GEMM_VARIANT

#define INV_VARIANT(T, Prefix, Suff)                                           \
  NUMBA_MLIR_MATH_RUNTIME_EXPORT int mkl_inv_##Suff(                           \
      Memref<2, T> *a, Memref<1, MKL_INT> *ipiv) {                             \
    return MKL_CALL(invImpl<T>, MKL_GETRF(Prefix), MKL_GETRI(Prefix), a,       \
                    ipiv);                                                     \
  }

INV_VARIANT(float, s, float32)
INV_VARIANT(double, d, float64)
INV_VARIANT(MKL_Complex8, c, complex64)
INV_VARIANT(MKL_Complex16, z, complex128)

#undef INV_VARIANT

#define SOLVE_VARIANT(T, Prefix, Suff)                                         \
  NUMBA_MLIR_MATH_RUNTIME_EXPORT int mkl_solve_##Suff(                         \
      Memref<2, T> *a, Memref<2, T> *b, Memref<1, MKL_INT> *ipiv) {            \
    return MKL_CALL(solveImpl<T>, MKL_GETSV(Prefix), a, b, ipiv);              \
  }

SOLVE_VARIANT(float, s, float32)
SOLVE_VARIANT(double, d, float64)
SOLVE_VARIANT(MKL_Complex8, c, complex64)
SOLVE_VARIANT(MKL_Complex16, z, complex128)

#undef SOLVE_VARIANT

#define CHOLESKY_VARIANT(T, Prefix, Suff)                                      \
  NUMBA_MLIR_MATH_RUNTIME_EXPORT int mkl_cholesky_##Suff(Memref<2, T> *a) {    \
    return MKL_CALL(choleskyImpl<T>, MKL_POTRF(Prefix), a);                    \
  }

CHOLESKY_VARIANT(float, s, float32)
CHOLESKY_VARIANT(double, d, float64)
CHOLESKY_VARIANT(MKL_Complex8, c, complex64)
CHOLESKY_VARIANT(MKL_Complex16, z, complex128)

#undef CHOLESKY_VARIANT

#define EIG_VARIANT_REAL(T, Prefix, Suff)                                      \
  NUMBA_MLIR_MATH_RUNTIME_EXPORT int mkl_eig_##Suff(                           \
      char jobvl, char jobvr, Memref<2, T> *a, Memref<1, T> *wr,               \
      Memref<1, T> *wi, Memref<2, T> *vl, Memref<2, T> *vr) {                  \
    return MKL_CALL(eigImplReal<T>, MKL_GEEV(Prefix), jobvl, jobvr, a, wr, wi, \
                    vl, vr);                                                   \
  }
#define EIG_VARIANT_COMPLEX(T, Prefix, Suff)                                   \
  NUMBA_MLIR_MATH_RUNTIME_EXPORT int mkl_eig_##Suff(                           \
      char jobvl, char jobvr, Memref<2, T> *a, Memref<1, T> *w,                \
      Memref<2, T> *vl, Memref<2, T> *vr) {                                    \
    return MKL_CALL(eigImplComplex<T>, MKL_GEEV(Prefix), jobvl, jobvr, a, w,   \
                    vl, vr);                                                   \
  }

EIG_VARIANT_REAL(float, s, float32)
EIG_VARIANT_REAL(double, d, float64)
EIG_VARIANT_COMPLEX(MKL_Complex8, c, complex64)
EIG_VARIANT_COMPLEX(MKL_Complex16, z, complex128)

#undef EIG_VARIANT_REAL
#undef EIG_VARIANT_COMPLEX

#define EIGH_VARIANT_REAL(T, TW, Prefix, Suff)                                 \
  NUMBA_MLIR_MATH_RUNTIME_EXPORT int mkl_eigh_##Suff(                          \
      char jobz, char uplo, Memref<2, T> *a, Memref<1, TW> *w) {               \
    return MKL_CALL(eighImpl<T>, MKL_SYEVD(Prefix), jobz, uplo, a, w);         \
  }

#define EIGH_VARIANT_COMPLEX(T, TW, Prefix, Suff)                              \
  NUMBA_MLIR_MATH_RUNTIME_EXPORT int mkl_eigh_##Suff(                          \
      char jobz, char uplo, Memref<2, T> *a, Memref<1, TW> *w) {               \
    return MKL_CALL(eighImpl<T>, MKL_HEEVD(Prefix), jobz, uplo, a, w);         \
  }

EIGH_VARIANT_REAL(float, float, s, float32)
EIGH_VARIANT_REAL(double, double, d, float64)
EIGH_VARIANT_COMPLEX(MKL_Complex8, float, c, complex64)
EIGH_VARIANT_COMPLEX(MKL_Complex16, double, z, complex128)

#undef EIGH_VARIANT_REAL
#undef EIGH_VARIANT_COMPLEX
}
