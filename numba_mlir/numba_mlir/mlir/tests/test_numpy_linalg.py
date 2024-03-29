# SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import numpy as np
from numba_mlir.mlir.utils import readenv
from numbers import Number, Integral

from .utils import njit_cached as njit

_linalg_dtypes = (np.float64, np.float32, np.complex128, np.complex64)


def vvsort(val, vec, size):
    for i in range(size):
        imax = i
        for j in range(i + 1, size):
            if np.abs(val[imax]) < np.abs(val[j]):
                imax = j

        temp = val[i]
        val[i] = val[imax]
        val[imax] = temp

        for k in range(size):
            temp = vec[k, i]
            vec[k, i] = vec[k, imax]
            vec[k, imax] = temp


@pytest.mark.parametrize("type", _linalg_dtypes)
@pytest.mark.parametrize("size", [2, 4, 8, 16])
def test_eig_arange(type, size):
    a = np.arange(size * size, dtype=type).reshape((size, size))
    symm_orig = (
        np.tril(a)
        + np.tril(a, -1).T
        + np.diag(np.full((size,), size * size, dtype=type))
    )
    symm = symm_orig.copy()
    dpnp_symm_orig = symm_orig.copy()
    dpnp_symm = symm_orig.copy()

    def py_func_val(s):
        return np.linalg.eig(s)[0]

    def py_func_vec(s):
        return np.linalg.eig(s)[1]

    jit_func_val = njit(py_func_val)
    jit_func_vec = njit(py_func_vec)

    dpnp_val, dpnp_vec = (jit_func_val(dpnp_symm), jit_func_vec(dpnp_symm))
    np_val, np_vec = (py_func_val(symm), py_func_vec(symm))

    # DPNP sort val/vec by abs value
    vvsort(dpnp_val, dpnp_vec, size)

    # NP sort val/vec by abs value
    vvsort(np_val, np_vec, size)

    # NP change sign of vectors
    for i in range(np_vec.shape[1]):
        if np_vec[0, i] * dpnp_vec[0, i] < 0:
            np_vec[:, i] = -np_vec[:, i]

    np.testing.assert_array_equal(symm_orig, symm)
    np.testing.assert_array_equal(dpnp_symm_orig, dpnp_symm)

    assert dpnp_val.dtype == np_val.dtype
    assert dpnp_vec.dtype == np_vec.dtype
    assert dpnp_val.shape == np_val.shape
    assert dpnp_vec.shape == np_vec.shape

    tol = 1e-03
    np.testing.assert_allclose(dpnp_val, np_val, rtol=tol, atol=tol)
    np.testing.assert_allclose(dpnp_vec, np_vec, rtol=tol, atol=tol)


def _sample_vector(n, dtype):
    # Be careful to generate only exactly representable float values,
    # to avoid rounding discrepancies between Numpy and Numba
    base = np.arange(n)
    if issubclass(dtype, np.complexfloating):
        return (base * (1 - 0.5j) + 2j).astype(dtype)
    else:
        return (base * 0.5 + 1).astype(dtype)


def _assert_contig_sanity(got, expected_contig):
    """
    This checks that in a computed result from numba (array, possibly tuple
    of arrays) all the arrays are contiguous in memory and that they are
    all at least one of "C_CONTIGUOUS" or "F_CONTIGUOUS". The computed
    result of the contiguousness is then compared against a hardcoded
    expected result.

    got: is the computed results from numba
    expected_contig: is "C" or "F" and is the expected type of
                    contiguousness across all input values
                    (and therefore tests).
    """

    if isinstance(got, tuple):
        # tuple present, check all results
        for a in got:
            _assert_contig_sanity(a, expected_contig)
    else:
        if not isinstance(got, Number):
            # else a single array is present
            c_contig = got.flags.c_contiguous
            f_contig = got.flags.f_contiguous

            # check that the result (possible set of) is at least one of
            # C or F contiguous.
            msg = "Results are not at least one of all C or F contiguous."
            assert c_contig | f_contig, msg

            msg = "Computed contiguousness does not match expected."
            if expected_contig == "C":
                assert c_contig, msg
            elif expected_contig == "F":
                assert f_contig, msg
            else:
                raise ValueError("Unknown contig")


def _assert_is_identity_matrix(got, rtol=None, atol=None):
    """
    Checks if a matrix is equal to the identity matrix.
    """
    # check it is square
    assert got.shape[-1] == got.shape[-2]
    # create identity matrix
    eye = np.eye(got.shape[-1], dtype=got.dtype)
    resolution = 5 * np.finfo(got.dtype).resolution
    if rtol is None:
        rtol = 10 * resolution
    if atol is None:
        atol = 100 * resolution  # zeros tend to be fuzzy
    # check it matches
    np.testing.assert_allclose(got, eye, rtol, atol)


def _sample_matrix(m, dtype, order):
    # pd. (positive definite) matrix has eigenvalues in Z+
    np.random.seed(0)  # repeatable seed
    A = np.random.rand(m, m)
    # orthonormal q needed to form up q^{-1}*D*q
    # no "orth()" in numpy
    q, _ = np.linalg.qr(A)
    L = np.arange(1, m + 1)  # some positive eigenvalues
    Q = np.dot(np.dot(q.T, np.diag(L)), q)  # construct
    Q = np.array(Q, dtype=dtype, order=order)  # sort out order/type
    return Q


def _specific_sample_matrix(size, dtype, order, rank=None, condition=None):
    """
    Provides a sample matrix with an optionally specified rank or condition
    number.

    size: (rows, columns), the dimensions of the returned matrix.
    dtype: the dtype for the returned matrix.
    order: the memory layout for the returned matrix, 'F' or 'C'.
    rank: the rank of the matrix, an integer value, defaults to full rank.
    condition: the condition number of the matrix (defaults to 1.)

    NOTE: Only one of rank or condition may be set.
    """

    # default condition
    d_cond = 1.0

    if len(size) != 2:
        raise ValueError("size must be a length 2 tuple.")

    if order not in ["F", "C"]:
        raise ValueError("order must be one of 'F' or 'C'.")

    if dtype not in [np.float32, np.float64, np.complex64, np.complex128]:
        raise ValueError("dtype must be a numpy floating point type.")

    if rank is not None and condition is not None:
        raise ValueError("Only one of rank or condition can be specified.")

    if condition is None:
        condition = d_cond

    if condition < 1:
        raise ValueError("Condition number must be >=1.")

    np.random.seed(0)  # repeatable seed
    m, n = size

    if m < 0 or n < 0:
        raise ValueError("Negative dimensions given for matrix shape.")

    minmn = min(m, n)
    if rank is None:
        rv = minmn
    else:
        if rank <= 0:
            raise ValueError("Rank must be greater than zero.")
        if not isinstance(rank, Integral):
            raise ValueError("Rank must an integer.")
        rv = rank
        if rank > minmn:
            raise ValueError("Rank given greater than full rank.")

    if m == 1 or n == 1:
        # vector, must be rank 1 (enforced above)
        # condition of vector is also 1
        if condition != d_cond:
            raise ValueError("Condition number was specified for a vector (always 1.).")
        maxmn = max(m, n)
        Q = _sample_vector(maxmn, dtype).reshape(m, n)
    else:
        # Build a sample matrix via combining SVD like inputs.

        # Create matrices of left and right singular vectors.
        # This could use Modified Gram-Schmidt and perhaps be quicker,
        # at present it uses QR decompositions to obtain orthonormal
        # matrices.
        tmp = _sample_vector(m * m, dtype).reshape(m, m)
        U, _ = np.linalg.qr(tmp)
        # flip the second array, else for m==n the identity matrix appears
        tmp = _sample_vector(n * n, dtype)[::-1].reshape(n, n)
        V, _ = np.linalg.qr(tmp)
        # create singular values.
        sv = np.linspace(d_cond, condition, rv)
        S = np.zeros((m, n))
        idx = np.nonzero(np.eye(m, n))
        S[idx[0][:rv], idx[1][:rv]] = sv
        Q = np.dot(np.dot(U, S), V.T)  # construct
        Q = np.array(Q, dtype=dtype, order=order)  # sort out order/type

    return Q


def _inv_checker(py_func, cfunc, a):
    expected = py_func(a)
    got = cfunc(a)
    # _assert_contig_sanity(got, "F")

    use_reconstruction = False

    # try strict
    try:
        np.testing.assert_array_almost_equal_nulp(got, expected, nulp=10)
    except AssertionError:
        # fall back to reconstruction
        use_reconstruction = True

    if use_reconstruction:
        rec = np.dot(got, a)
        _assert_is_identity_matrix(rec)


@pytest.mark.parametrize("n", [0, 10, 107])
@pytest.mark.parametrize("dtype", _linalg_dtypes)
@pytest.mark.parametrize("order", "CF")
def test_linalg_inv(n, dtype, order):
    def py_func(a):
        return np.linalg.inv(a)

    jit_func = njit(py_func)

    a = _specific_sample_matrix((n, n), dtype, order)
    _inv_checker(py_func, jit_func, a)


def _cholesky_checker(py_func, cfunc, a):
    expected = py_func(a)
    got = cfunc(a)
    use_reconstruction = False
    # check that the computed results are contig and in the same way
    # self.assert_contig_sanity(got, "C")

    # try strict
    try:
        np.testing.assert_array_almost_equal_nulp(got, expected, nulp=10)
    except AssertionError:
        # fall back to reconstruction
        use_reconstruction = True

    # try via reconstruction
    if use_reconstruction:
        rec = np.dot(got, np.conj(got.T))
        resolution = 5 * np.finfo(a.dtype).resolution
        np.testing.assert_allclose(a, rec, rtol=resolution, atol=resolution)


@pytest.mark.parametrize("n", [0, 10, 107])
@pytest.mark.parametrize("dtype", _linalg_dtypes)
@pytest.mark.parametrize("order", "CF")
def test_linalg_cholesky(n, dtype, order):
    def py_func(a):
        return np.linalg.cholesky(a)

    jit_func = njit(py_func)

    a = _sample_matrix(n, dtype, order)

    _cholesky_checker(py_func, jit_func, a)


def _solve_checker(py_func, cfunc, a, b):
    expected = py_func(a, b)
    got = cfunc(a, b)

    # check that the computed results are contig and in the same way
    # self.assert_contig_sanity(got, "F")

    use_reconstruction = False
    # try plain match of the result first
    try:
        np.testing.assert_array_almost_equal_nulp(got, expected, nulp=10)
    except AssertionError:
        # plain match failed, test by reconstruction
        use_reconstruction = True

    # If plain match fails then reconstruction is used,
    # this checks that AX ~= B.
    # Plain match can fail due to numerical fuzziness associated
    # with system size and conditioning, or more simply from
    # numpy using double precision routines for computation that
    # could be done in single precision (which is what numba does).
    # Therefore minor differences in results can appear due to
    # e.g. numerical roundoff being different between two precisions.
    if use_reconstruction:
        # check they are dimensionally correct
        assert got.shape == expected.shape

        # check AX=B
        rec = np.dot(a, got)
        resolution = np.finfo(a.dtype).resolution
        np.testing.assert_allclose(
            b,
            rec,
            rtol=10 * resolution,
            atol=100 * resolution,  # zeros tend to be fuzzy
        )


@pytest.mark.parametrize("a_size", [(1, 1), (3, 3), (7, 7)])
@pytest.mark.parametrize("b_size", [None, 1, 13])
@pytest.mark.parametrize("dtype", _linalg_dtypes)
@pytest.mark.parametrize("a_order", "CF")
@pytest.mark.parametrize("b_order", "CF")
def test_linalg_solve(a_size, b_size, dtype, a_order, b_order):
    def py_func(a, b):
        return np.linalg.solve(a, b)

    jit_func = njit(py_func)

    a = _specific_sample_matrix(a_size, dtype, a_order)
    n = a.shape[0]
    if b_size is None:
        b = _specific_sample_matrix((n, 1), dtype, b_order).reshape((n,))
    else:
        b = _specific_sample_matrix((n, b_size), dtype, b_order)

    _solve_checker(py_func, jit_func, a, b)


def test_linalg_solve_empty():
    def py_func(a, b):
        return np.linalg.solve(a, b)

    jit_func = njit(py_func)

    a = np.empty((0, 0))
    b = np.empty((0,))

    _solve_checker(py_func, jit_func, a, b)


def _eig_check_worker(cfunc, name, expected_res_len):
    def check(*args):
        expected = cfunc.py_func(*args)
        got = cfunc(*args)
        a = args[0]
        # check that the returned tuple is same length
        assert len(expected) == len(got)
        # and that dimension is correct
        res_is_tuple = False
        if isinstance(got, tuple):
            res_is_tuple = True
            assert len(got) == expected_res_len
        else:  # its an array
            assert got.ndim == expected_res_len

        use_reconstruction = False
        # try plain match of each array to np first
        for k in range(len(expected)):
            try:
                np.testing.assert_array_almost_equal_nulp(got[k], expected[k], nulp=10)
            except AssertionError:
                # plain match failed, test by reconstruction
                use_reconstruction = True

        # If plain match fails then reconstruction is used.
        # this checks that A*V ~== V*diag(W)
        # i.e. eigensystem ties out
        # this is required as numpy uses only double precision lapack
        # routines and computation of eigenvectors is numerically
        # sensitive, numba uses the type specific routines therefore
        # sometimes comes out with a different (but entirely
        # valid) answer (eigenvectors are not unique etc.).
        # This is only applicable if eigenvectors are computed
        # along with eigenvalues i.e. result is a tuple.
        resolution = 10 * np.finfo(a.dtype).resolution
        # resolution = 5 * np.finfo(a.dtype).resolution
        if use_reconstruction:
            if res_is_tuple:
                w, v = got
                # modify 'a' if hermitian eigensystem functionality is
                # being tested. 'L' for use lower part is default and
                # the only thing used at present so we conjugate transpose
                # the lower part into the upper for use in the
                # reconstruction. By construction the sample matrix is
                # tridiag so this is just a question of copying the lower
                # diagonal into the upper and conjugating on the way.
                if name[-1] == "h":
                    idxl = np.nonzero(np.eye(a.shape[0], a.shape[1], -1))
                    idxu = np.nonzero(np.eye(a.shape[0], a.shape[1], 1))
                    cfunc(*args)
                    # upper idx must match lower for default uplo="L"
                    # if complex, conjugate
                    a[idxu] = np.conj(a[idxl])
                    # also, only the real part of the diagonals is
                    # considered in the calculation so the imag is zeroed
                    # out for the purposes of use in reconstruction.
                    a[np.diag_indices(a.shape[0])] = np.real(np.diag(a))

                lhs = np.dot(a, v)
                rhs = np.dot(v, np.diag(w))

                np.testing.assert_allclose(
                    lhs.real, rhs.real, rtol=resolution, atol=resolution
                )
                if np.iscomplexobj(v):
                    np.testing.assert_allclose(
                        lhs.imag, rhs.imag, rtol=resolution, atol=resolution
                    )
            else:
                # This isn't technically reconstruction but is here to
                # deal with that the order of the returned eigenvalues
                # may differ in the case of routines just returning
                # eigenvalues and there's no true reconstruction
                # available with which to perform a check.
                np.testing.assert_allclose(
                    np.sort(expected), np.sort(got), rtol=resolution, atol=resolution
                )

    return check


@pytest.mark.parametrize("n", [0, 10])
@pytest.mark.parametrize("dtype", _linalg_dtypes)
@pytest.mark.parametrize("order", "CF")
@pytest.mark.parametrize("func", ["eig", "eigvals", "eigh", "eigvalsh"])
def test_linalg_eig(n, dtype, order, func):
    expected_len = 1 if "vals" in func else 2
    func = eval("np.linalg." + func)

    def py_func(a):
        return func(a)

    jit_func = njit(py_func)

    checker = _eig_check_worker(jit_func, "eig", expected_len)

    a = _sample_matrix(n, dtype, order)
    checker(a)
