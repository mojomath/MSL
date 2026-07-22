# SPDX-License-Identifier: GPL-3.0-or-later

from std.testing import TestSuite
from std.math import abs

from msl.linalg import (
    lu_decomp,
    lu_solve,
    lu_det,
    lu_invert,
    cholesky_decomp,
    cholesky_solve,
    qr_decomp,
    qr_solve,
    eigen_jacobi,
)


def tol(x: Float64, expected: Float64, t: Float64) -> Bool:
    return abs(x - expected) < t


def test_lu_solve_and_det() raises:
    # A = [[2,1],[1,3]], b = [3,5] -> x = [0.8, 1.4]
    var a = alloc[Float64](4)
    a[0] = 2.0
    a[1] = 1.0
    a[2] = 1.0
    a[3] = 3.0

    var piv = alloc[Int](2)
    var signum = lu_decomp(a, 2, 2, piv)
    assert signum != 0

    var b = alloc[Float64](2)
    b[0] = 3.0
    b[1] = 5.0
    var x = alloc[Float64](2)
    lu_solve(a, 2, 2, piv, b, x)

    assert tol(x[0], 0.8, 1e-10)
    assert tol(x[1], 1.4, 1e-10)

    var det = lu_det(a, 2, 2, signum)
    assert tol(det, 5.0, 1e-10)
    print("test_lu_solve_and_det: PASSED")


def test_lu_invert() raises:
    # A = [[4,3],[6,3]] -> A^-1 = [[-0.5, 0.5],[1, -2/3]]
    var a = alloc[Float64](4)
    a[0] = 4.0
    a[1] = 3.0
    a[2] = 6.0
    a[3] = 3.0

    var piv = alloc[Int](2)
    _ = lu_decomp(a, 2, 2, piv)

    var inv = alloc[Float64](4)
    var work = alloc[Float64](2)
    lu_invert(a, 2, 2, piv, inv, 2, work)

    assert tol(inv[0], -0.5, 1e-10)
    assert tol(inv[1], 0.5, 1e-10)
    assert tol(inv[2], 1.0, 1e-10)
    assert tol(inv[3], -2.0 / 3.0, 1e-10)
    print("test_lu_invert: PASSED")


def test_cholesky_solve() raises:
    # A = [[4,2],[2,3]] (SPD), b = [6,5]
    # Solve directly: 4x + 2y = 6, 2x + 3y = 5 -> x=1, y=1
    var a = alloc[Float64](4)
    a[0] = 4.0
    a[1] = 2.0
    a[2] = 2.0
    a[3] = 3.0

    var stat = cholesky_decomp(a, 2, 2)
    assert stat == 0

    var b = alloc[Float64](2)
    b[0] = 6.0
    b[1] = 5.0
    var x = alloc[Float64](2)
    cholesky_solve(a, 2, 2, b, x)

    assert tol(x[0], 1.0, 1e-9)
    assert tol(x[1], 1.0, 1e-9)
    print("test_cholesky_solve: PASSED")


def test_cholesky_not_positive_definite() raises:
    # A = [[1,2],[2,1]] is symmetric but not positive-definite
    var a = alloc[Float64](4)
    a[0] = 1.0
    a[1] = 2.0
    a[2] = 2.0
    a[3] = 1.0

    var stat = cholesky_decomp(a, 2, 2)
    assert stat != 0
    print("test_cholesky_not_positive_definite: PASSED")


def test_qr_solve() raises:
    # Same system as LU test: A = [[2,1],[1,3]], b = [3,5] -> x = [0.8, 1.4]
    var a = alloc[Float64](4)
    a[0] = 2.0
    a[1] = 1.0
    a[2] = 1.0
    a[3] = 3.0

    var tau = alloc[Float64](2)
    qr_decomp(a, 2, 2, tau)

    var b = alloc[Float64](2)
    b[0] = 3.0
    b[1] = 5.0
    var x = alloc[Float64](2)
    qr_solve(a, 2, 2, tau, b, x)

    assert tol(x[0], 0.8, 1e-9)
    assert tol(x[1], 1.4, 1e-9)
    print("test_qr_solve: PASSED")


def test_eigen_jacobi_symmetric() raises:
    # A = [[2,1],[1,2]] -> eigenvalues 1, 3
    var a = alloc[Float64](4)
    a[0] = 2.0
    a[1] = 1.0
    a[2] = 1.0
    a[3] = 2.0

    var eval = alloc[Float64](2)
    var evec = alloc[Float64](4)
    _ = eigen_jacobi(a, 2, 2, eval, evec, 2)

    var lo = eval[0] if eval[0] < eval[1] else eval[1]
    var hi = eval[0] if eval[0] > eval[1] else eval[1]
    assert tol(lo, 1.0, 1e-9)
    assert tol(hi, 3.0, 1e-9)

    # eigenvectors should be orthonormal columns: check ||v0|| == 1
    var norm0_sq = evec[0] * evec[0] + evec[2] * evec[2]
    assert tol(norm0_sq, 1.0, 1e-9)
    print("test_eigen_jacobi_symmetric: PASSED")


def test_eigen_jacobi_diagonal() raises:
    # already-diagonal matrix: eigenvalues are the diagonal entries
    var a = alloc[Float64](9)
    for i in range(9):
        a[i] = 0.0
    a[0] = 5.0
    a[4] = 2.0
    a[8] = 9.0

    var eval = alloc[Float64](3)
    var evec = alloc[Float64](9)
    _ = eigen_jacobi(a, 3, 3, eval, evec, 3)

    var found5 = False
    var found2 = False
    var found9 = False
    for i in range(3):
        if tol(eval[i], 5.0, 1e-9):
            found5 = True
        if tol(eval[i], 2.0, 1e-9):
            found2 = True
        if tol(eval[i], 9.0, 1e-9):
            found9 = True
    assert found5 and found2 and found9
    print("test_eigen_jacobi_diagonal: PASSED")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("All linalg tests PASSED")
