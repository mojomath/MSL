# SPDX-License-Identifier: GPL-3.0-or-later

from std.testing import TestSuite
from std.math import abs

from msl.poly import (
    poly_eval,
    poly_eval_derivs,
    poly_dd_init,
    poly_dd_eval,
    poly_dd_taylor,
    poly_solve_quadratic,
    poly_solve_cubic,
)


def tol(x: Float64, expected: Float64, t: Float64) -> Bool:
    return abs(x - expected) < t


def test_poly_eval() raises:
    # p(x) = 1 + 2x + 3x^2
    var c = alloc[Float64](3)
    c[0] = 1.0
    c[1] = 2.0
    c[2] = 3.0

    assert tol(poly_eval(c, 3, 0.0), 1.0, 1e-12)
    assert tol(poly_eval(c, 3, 1.0), 6.0, 1e-12)
    assert tol(poly_eval(c, 3, 2.0), 17.0, 1e-12)
    print("test_poly_eval: PASSED")


def test_poly_eval_derivs() raises:
    # p(x) = 1 + 2x + 3x^2 + 4x^3
    # p'(x) = 2 + 6x + 12x^2
    # p''(x) = 6 + 24x
    # p'''(x) = 24
    var c = alloc[Float64](4)
    c[0] = 1.0
    c[1] = 2.0
    c[2] = 3.0
    c[3] = 4.0

    var res = alloc[Float64](4)
    poly_eval_derivs(c, 4, 2.0, res, 4)

    assert tol(res[0], 1.0 + 2.0 * 2.0 + 3.0 * 4.0 + 4.0 * 8.0, 1e-9)
    assert tol(res[1], 2.0 + 6.0 * 2.0 + 12.0 * 4.0, 1e-9)
    assert tol(res[2], 6.0 + 24.0 * 2.0, 1e-9)
    assert tol(res[3], 24.0, 1e-9)
    print("test_poly_eval_derivs: PASSED")


def test_poly_dd() raises:
    # interpolate y = x^2 at x = 0, 1, 2
    var x = alloc[Float64](3)
    var y = alloc[Float64](3)
    x[0] = 0.0
    x[1] = 1.0
    x[2] = 2.0
    y[0] = 0.0
    y[1] = 1.0
    y[2] = 4.0

    var dd = alloc[Float64](3)
    poly_dd_init(dd, x, y, 3)

    assert tol(poly_dd_eval(dd, x, 3, 0.0), 0.0, 1e-9)
    assert tol(poly_dd_eval(dd, x, 3, 1.0), 1.0, 1e-9)
    assert tol(poly_dd_eval(dd, x, 3, 2.0), 4.0, 1e-9)
    assert tol(poly_dd_eval(dd, x, 3, 3.0), 9.0, 1e-9)

    var c = alloc[Float64](3)
    var w = alloc[Float64](3)
    poly_dd_taylor(c, 0.0, dd, x, 3, w)
    # Taylor expansion about 0 of x^2 should just be [0, 0, 1]
    assert tol(c[0], 0.0, 1e-9)
    assert tol(c[1], 0.0, 1e-9)
    assert tol(c[2], 1.0, 1e-9)
    print("test_poly_dd: PASSED")


def test_solve_quadratic() raises:
    # x^2 - 3x + 2 = 0 -> roots 1, 2
    var r = poly_solve_quadratic(1.0, -3.0, 2.0)
    assert r.nroots == 2
    assert tol(r.x0, 1.0, 1e-12)
    assert tol(r.x1, 2.0, 1e-12)

    # x^2 + 1 = 0 -> no real roots
    var r2 = poly_solve_quadratic(1.0, 0.0, 1.0)
    assert r2.nroots == 0

    # x^2 - 2x + 1 = 0 -> double root at 1
    var r3 = poly_solve_quadratic(1.0, -2.0, 1.0)
    assert r3.nroots == 2
    assert tol(r3.x0, 1.0, 1e-12)
    assert tol(r3.x1, 1.0, 1e-12)

    # linear case: 2x - 4 = 0 -> x = 2
    var r4 = poly_solve_quadratic(0.0, 2.0, -4.0)
    assert r4.nroots == 1
    assert tol(r4.x0, 2.0, 1e-12)
    print("test_solve_quadratic: PASSED")


def test_solve_cubic() raises:
    # x^3 - 6x^2 + 11x - 6 = 0 -> roots 1, 2, 3
    var r = poly_solve_cubic(-6.0, 11.0, -6.0)
    assert r.nroots == 3
    assert tol(r.x0, 1.0, 1e-9)
    assert tol(r.x1, 2.0, 1e-9)
    assert tol(r.x2, 3.0, 1e-9)

    # x^3 - 3x^2 + 3x - 1 = (x-1)^3 = 0 -> triple root at 1
    var r2 = poly_solve_cubic(-3.0, 3.0, -1.0)
    assert r2.nroots == 3
    assert tol(r2.x0, 1.0, 1e-6)
    assert tol(r2.x1, 1.0, 1e-6)
    assert tol(r2.x2, 1.0, 1e-6)

    # x^3 - 1 = 0 -> one real root at 1
    var r3 = poly_solve_cubic(0.0, 0.0, -1.0)
    assert r3.nroots == 1
    assert tol(r3.x0, 1.0, 1e-9)
    print("test_solve_cubic: PASSED")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("All poly tests PASSED")
