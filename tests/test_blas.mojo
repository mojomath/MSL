from std.testing import TestSuite
from std.math import abs, sqrt

from msl.vector import Vector
from msl.matrix import Matrix
from msl.blas import (
    blas_dot, blas_nrm2, blas_asum, blas_axpy,
    blas_scal, blas_copy, blas_swap, blas_gemv, blas_gemm,
)


def tol(x: Float64, expected: Float64, t: Float64) -> Bool:
    return abs(x - expected) < t


# ===----------------------------------------------------------------------=== #
# Level 1
# ===----------------------------------------------------------------------=== #


def test_blas_dot() raises:
    var x = Vector(3, initialize=True)
    var y = Vector(3, initialize=True)
    x[0] = 1.0; x[1] = 2.0; x[2] = 3.0
    y[0] = 4.0; y[1] = 5.0; y[2] = 6.0
    var d = blas_dot(x, y)
    assert tol(d, 32.0, 1e-14)
    print("test_blas_dot: PASSED")


def test_blas_nrm2() raises:
    var x = Vector(3, initialize=True)
    x[0] = 3.0; x[1] = 4.0; x[2] = 0.0
    var n = blas_nrm2(x)
    assert tol(n, 5.0, 1e-14)
    print("test_blas_nrm2: PASSED")


def test_blas_asum() raises:
    var x = Vector(3, initialize=True)
    x[0] = -1.0; x[1] = 2.0; x[2] = -3.0
    var s = blas_asum(x)
    assert tol(s, 6.0, 1e-14)
    print("test_blas_asum: PASSED")


def test_blas_axpy() raises:
    var x = Vector(3, initialize=True)
    var y = Vector(3, initialize=True)
    x[0] = 1.0; x[1] = 2.0; x[2] = 3.0
    y[0] = 10.0; y[1] = 10.0; y[2] = 10.0
    blas_axpy(2.0, x, y)
    assert tol(y[0], 12.0, 1e-14) and tol(y[1], 14.0, 1e-14) and tol(y[2], 16.0, 1e-14)
    print("test_blas_axpy: PASSED")


def test_blas_scal() raises:
    var x = Vector(3, initialize=True)
    x[0] = 1.0; x[1] = 2.0; x[2] = 3.0
    blas_scal(3.0, x)
    assert tol(x[0], 3.0, 1e-14) and tol(x[1], 6.0, 1e-14) and tol(x[2], 9.0, 1e-14)
    print("test_blas_scal: PASSED")


def test_blas_copy() raises:
    var x = Vector(3, initialize=True)
    var y = Vector(3, initialize=True)
    x[0] = 7.0; x[1] = 8.0; x[2] = 9.0
    blas_copy(x, y)
    assert tol(y[0], 7.0, 1e-14) and tol(y[1], 8.0, 1e-14) and tol(y[2], 9.0, 1e-14)
    print("test_blas_copy: PASSED")


def test_blas_swap() raises:
    var x = Vector(3, initialize=True)
    var y = Vector(3, initialize=True)
    x[0] = 1.0; x[1] = 2.0; x[2] = 3.0
    y[0] = 4.0; y[1] = 5.0; y[2] = 6.0
    blas_swap(x, y)
    assert tol(x[0], 4.0, 1e-14) and tol(y[0], 1.0, 1e-14)
    print("test_blas_swap: PASSED")


def test_blas_dot_size_mismatch() raises:
    var x = Vector(3, initialize=True)
    var y = Vector(4, initialize=True)
    var d = blas_dot(x, y)
    assert d == 0.0  # silently returns 0 on mismatch
    print("test_blas_dot_size_mismatch: PASSED")


# ===----------------------------------------------------------------------=== #
# Level 2
# ===----------------------------------------------------------------------=== #


def test_blas_gemv() raises:
    # A = [[1,2],[3,4]], x = [1,1], y = alpha*A*x + beta*y = [3, 7]
    var a = Matrix(2, 2, initialize=True)
    a[0, 0] = 1.0; a[0, 1] = 2.0
    a[1, 0] = 3.0; a[1, 1] = 4.0
    var x = Vector(2, initialize=True)
    x[0] = 1.0; x[1] = 1.0
    var y = Vector(2, initialize=True)
    blas_gemv(a, x, y)
    assert tol(y[0], 3.0, 1e-14) and tol(y[1], 7.0, 1e-14)
    print("test_blas_gemv: PASSED")


def test_blas_gemv_transposed() raises:
    # A^T * x = [[1,3],[2,4]] * [1,1] = [4, 6]
    var a = Matrix(2, 2, initialize=True)
    a[0, 0] = 1.0; a[0, 1] = 2.0
    a[1, 0] = 3.0; a[1, 1] = 4.0
    var x = Vector(2, initialize=True)
    x[0] = 1.0; x[1] = 1.0
    var y = Vector(2, initialize=True)
    blas_gemv(a, x, y, trans=True)
    assert tol(y[0], 4.0, 1e-14) and tol(y[1], 6.0, 1e-14)
    print("test_blas_gemv_transposed: PASSED")


# ===----------------------------------------------------------------------=== #
# Level 3
# ===----------------------------------------------------------------------=== #


def test_blas_gemm() raises:
    # [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
    var a = Matrix(2, 2, initialize=True)
    a[0, 0] = 1.0; a[0, 1] = 2.0; a[1, 0] = 3.0; a[1, 1] = 4.0
    var b = Matrix(2, 2, initialize=True)
    b[0, 0] = 5.0; b[0, 1] = 6.0; b[1, 0] = 7.0; b[1, 1] = 8.0
    var c = Matrix(2, 2, initialize=True)
    blas_gemm(a, b, c)
    assert tol(c[0, 0], 19.0, 1e-12) and tol(c[0, 1], 22.0, 1e-12)
    assert tol(c[1, 0], 43.0, 1e-12) and tol(c[1, 1], 50.0, 1e-12)
    print("test_blas_gemm: PASSED")


def test_blas_gemm_rectangular() raises:
    # (2x3) @ (3x2) → (2x2)
    var a = Matrix(2, 3, initialize=True)
    a[0, 0] = 1.0; a[0, 1] = 2.0; a[0, 2] = 3.0
    a[1, 0] = 4.0; a[1, 1] = 5.0; a[1, 2] = 6.0
    var b = Matrix(3, 2, initialize=True)
    b[0, 0] = 7.0; b[0, 1] = 8.0
    b[1, 0] = 9.0; b[1, 1] = 10.0
    b[2, 0] = 11.0; b[2, 1] = 12.0
    var c = Matrix(2, 2, initialize=True)
    blas_gemm(a, b, c)
    assert tol(c[0, 0], 58.0, 1e-12) and tol(c[0, 1], 64.0, 1e-12)
    assert tol(c[1, 0], 139.0, 1e-12) and tol(c[1, 1], 154.0, 1e-12)
    print("test_blas_gemm_rectangular: PASSED")


def test_blas_gemm_alpha_beta() raises:
    # C = 2*A*B + 3*C
    var a = Matrix(2, 2, initialize=True)
    a[0, 0] = 1.0; a[0, 1] = 0.0; a[1, 0] = 0.0; a[1, 1] = 1.0
    var b = Matrix(2, 2, initialize=True)
    b[0, 0] = 1.0; b[0, 1] = 0.0; b[1, 0] = 0.0; b[1, 1] = 1.0
    var c = Matrix(2, 2, initialize=True)
    c[0, 0] = 1.0; c[0, 1] = 0.0; c[1, 0] = 0.0; c[1, 1] = 1.0
    # C = 2*I*I + 3*I = 5*I
    blas_gemm(a, b, c, alpha=2.0, beta=3.0)
    assert tol(c[0, 0], 5.0, 1e-12) and tol(c[1, 1], 5.0, 1e-12)
    assert tol(c[0, 1], 0.0, 1e-12) and tol(c[1, 0], 0.0, 1e-12)
    print("test_blas_gemm_alpha_beta: PASSED")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("All BLAS tests PASSED")
