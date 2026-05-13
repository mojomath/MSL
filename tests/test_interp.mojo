from std.testing import TestSuite
from std.math import abs

from msl.interpolation import LinearInterp, CubicSpline, AkimaSpline
from msl.core.const import MSL_DBL_EPSILON

comptime MutExt = MutExternalOrigin


def tolerance(x: Float64, expected: Float64, tol: Float64) -> Bool:
    return abs(x - expected) < tol


def _make_quadratic() -> Tuple[UnsafePointer[Float64, MutExt], UnsafePointer[Float64, MutExt]]:
    """x=[0,1,2,3,4], y=x^2=[0,1,4,9,16]"""
    var xa = alloc[Float64](5)
    var ya = alloc[Float64](5)
    xa[0] = 0.0; xa[1] = 1.0; xa[2] = 2.0; xa[3] = 3.0; xa[4] = 4.0
    ya[0] = 0.0; ya[1] = 1.0; ya[2] = 4.0; ya[3] = 9.0; ya[4] = 16.0
    return (xa, ya)


def _make_cubic() -> Tuple[UnsafePointer[Float64, MutExt], UnsafePointer[Float64, MutExt]]:
    """x=[0,1,2,3,4], y=x^3=[0,1,8,27,64]"""
    var xa = alloc[Float64](5)
    var ya = alloc[Float64](5)
    xa[0] = 0.0; xa[1] = 1.0; xa[2] = 2.0; xa[3] = 3.0; xa[4] = 4.0
    ya[0] = 0.0; ya[1] = 1.0; ya[2] = 8.0; ya[3] = 27.0; ya[4] = 64.0
    return (xa, ya)


# ===----------------------------------------------------------------------=== #
# LinearInterp
# ===----------------------------------------------------------------------=== #


def test_linear_exact_nodes() raises:
    var xa: UnsafePointer[Float64, MutExt]
    var ya: UnsafePointer[Float64, MutExt]
    xa, ya = _make_quadratic()
    var interp = LinearInterp(xa, ya, 5)
    for i in range(5):
        var r = interp.eval(xa[i])
        assert r.errno == 0 and tolerance(r.val, ya[i], 1e-14)
    xa.free(); ya.free()
    print("test_linear_exact_nodes: PASSED")


def test_linear_midpoint() raises:
    var xa: UnsafePointer[Float64, MutExt]
    var ya: UnsafePointer[Float64, MutExt]
    xa, ya = _make_quadratic()
    var interp = LinearInterp(xa, ya, 5)
    # midpoint [1,2]: x=1.5 → (1+4)/2 = 2.5
    var r = interp.eval(1.5)
    assert r.errno == 0 and tolerance(r.val, 2.5, 1e-14)
    xa.free(); ya.free()
    print("test_linear_midpoint: PASSED")


def test_linear_deriv() raises:
    var xa: UnsafePointer[Float64, MutExt]
    var ya: UnsafePointer[Float64, MutExt]
    xa, ya = _make_quadratic()
    var interp = LinearInterp(xa, ya, 5)
    # slope on [1,2]: (4-1)/(2-1) = 3
    var r = interp.deriv(1.5)
    assert r.errno == 0 and tolerance(r.val, 3.0, 1e-14)
    xa.free(); ya.free()
    print("test_linear_deriv: PASSED")


def test_linear_deriv2_zero() raises:
    var xa: UnsafePointer[Float64, MutExt]
    var ya: UnsafePointer[Float64, MutExt]
    xa, ya = _make_quadratic()
    var interp = LinearInterp(xa, ya, 5)
    var r = interp.deriv2(1.5)
    assert r.errno == 0 and r.val == 0.0
    xa.free(); ya.free()
    print("test_linear_deriv2_zero: PASSED")


def test_linear_integral() raises:
    var xa: UnsafePointer[Float64, MutExt]
    var ya: UnsafePointer[Float64, MutExt]
    xa, ya = _make_quadratic()
    var interp = LinearInterp(xa, ya, 5)
    # trapezoid sum = (0+1)/2 + (1+4)/2 + (4+9)/2 + (9+16)/2 = 22
    var r = interp.integral(0.0, 4.0)
    assert r.errno == 0 and tolerance(r.val, 22.0, 1e-12)
    xa.free(); ya.free()
    print("test_linear_integral: PASSED")


def test_linear_domain_error() raises:
    var xa: UnsafePointer[Float64, MutExt]
    var ya: UnsafePointer[Float64, MutExt]
    xa, ya = _make_quadratic()
    var interp = LinearInterp(xa, ya, 5)
    assert interp.eval(-1.0).errno != 0
    assert interp.eval(5.0).errno != 0
    xa.free(); ya.free()
    print("test_linear_domain_error: PASSED")


# ===----------------------------------------------------------------------=== #
# CubicSpline
# ===----------------------------------------------------------------------=== #


def test_cspline_exact_nodes() raises:
    var xa: UnsafePointer[Float64, MutExt]
    var ya: UnsafePointer[Float64, MutExt]
    xa, ya = _make_cubic()
    var interp = CubicSpline(xa, ya, 5)
    for i in range(5):
        var r = interp.eval(xa[i])
        assert r.errno == 0 and tolerance(r.val, ya[i], 1e-10)
    xa.free(); ya.free()
    print("test_cspline_exact_nodes: PASSED")


def test_cspline_interior_cubic() raises:
    var xa: UnsafePointer[Float64, MutExt]
    var ya: UnsafePointer[Float64, MutExt]
    xa, ya = _make_cubic()
    var interp = CubicSpline(xa, ya, 5)
    # x=1.5 → x^3=3.375; spline should be very close
    var r = interp.eval(1.5)
    assert r.errno == 0 and tolerance(r.val, 3.375, 0.1)
    xa.free(); ya.free()
    print("test_cspline_interior_cubic: PASSED")


def test_cspline_deriv() raises:
    var xa: UnsafePointer[Float64, MutExt]
    var ya: UnsafePointer[Float64, MutExt]
    xa, ya = _make_quadratic()
    var interp = CubicSpline(xa, ya, 5)
    # d(x^2)/dx at x=2 = 4
    var r = interp.deriv(2.0)
    assert r.errno == 0 and tolerance(r.val, 4.0, 0.1)
    xa.free(); ya.free()
    print("test_cspline_deriv: PASSED")


def test_cspline_integral() raises:
    var xa: UnsafePointer[Float64, MutExt]
    var ya: UnsafePointer[Float64, MutExt]
    xa, ya = _make_quadratic()
    var interp = CubicSpline(xa, ya, 5)
    # integral x^2 from 0 to 4 = 64/3 ≈ 21.333
    var r = interp.integral(0.0, 4.0)
    assert r.errno == 0 and tolerance(r.val, 64.0 / 3.0, 0.1)
    xa.free(); ya.free()
    print("test_cspline_integral: PASSED")


def test_cspline_domain_error() raises:
    var xa: UnsafePointer[Float64, MutExt]
    var ya: UnsafePointer[Float64, MutExt]
    xa, ya = _make_quadratic()
    var interp = CubicSpline(xa, ya, 5)
    assert interp.eval(-1.0).errno != 0
    assert interp.eval(5.0).errno != 0
    xa.free(); ya.free()
    print("test_cspline_domain_error: PASSED")


# ===----------------------------------------------------------------------=== #
# AkimaSpline
# ===----------------------------------------------------------------------=== #


def test_akima_exact_nodes() raises:
    var xa: UnsafePointer[Float64, MutExt]
    var ya: UnsafePointer[Float64, MutExt]
    xa, ya = _make_cubic()
    var interp = AkimaSpline(xa, ya, 5)
    for i in range(5):
        var r = interp.eval(xa[i])
        assert r.errno == 0 and tolerance(r.val, ya[i], 1e-10)
    xa.free(); ya.free()
    print("test_akima_exact_nodes: PASSED")


def test_akima_interior() raises:
    var xa: UnsafePointer[Float64, MutExt]
    var ya: UnsafePointer[Float64, MutExt]
    xa, ya = _make_cubic()
    var interp = AkimaSpline(xa, ya, 5)
    var r = interp.eval(1.5)
    assert r.errno == 0 and tolerance(r.val, 3.375, 0.1)
    xa.free(); ya.free()
    print("test_akima_interior: PASSED")


def test_akima_deriv() raises:
    var xa: UnsafePointer[Float64, MutExt]
    var ya: UnsafePointer[Float64, MutExt]
    xa, ya = _make_quadratic()
    var interp = AkimaSpline(xa, ya, 5)
    var r = interp.deriv(2.0)
    assert r.errno == 0 and tolerance(r.val, 4.0, 0.1)
    xa.free(); ya.free()
    print("test_akima_deriv: PASSED")


def test_akima_integral() raises:
    var xa: UnsafePointer[Float64, MutExt]
    var ya: UnsafePointer[Float64, MutExt]
    xa, ya = _make_quadratic()
    var interp = AkimaSpline(xa, ya, 5)
    var r = interp.integral(0.0, 4.0)
    assert r.errno == 0 and tolerance(r.val, 64.0 / 3.0, 0.5)
    xa.free(); ya.free()
    print("test_akima_integral: PASSED")


def test_akima_domain_error() raises:
    var xa: UnsafePointer[Float64, MutExt]
    var ya: UnsafePointer[Float64, MutExt]
    xa, ya = _make_quadratic()
    var interp = AkimaSpline(xa, ya, 5)
    assert interp.eval(-1.0).errno != 0
    assert interp.eval(5.0).errno != 0
    xa.free(); ya.free()
    print("test_akima_domain_error: PASSED")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("All interpolation tests PASSED")
