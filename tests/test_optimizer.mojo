from std.testing import TestSuite
from std.math import abs, sin, cos, sqrt, exp

from msl.optimizer import root_bisect, root_brent, root_newton, root_secant
from msl.optimizer import min_brent, min_golden


def tol(x: Float64, expected: Float64, t: Float64) -> Bool:
    return abs(x - expected) < t


# ===----------------------------------------------------------------------=== #
# root_bisect
# ===----------------------------------------------------------------------=== #


def test_bisect_sin() raises:
    # sin(x) = 0 near pi
    def fn_(x: Float64) capturing -> Float64:
        return sin(x)

    var r = root_bisect[fn_](3.0, 4.0)
    assert r.success and tol(r.root, 3.141592653589793, 1e-8)
    print("test_bisect_sin: PASSED")


def test_bisect_poly() raises:
    # x^3 - x - 1 = 0, root near 1.3247
    def fn_(x: Float64) capturing -> Float64:
        return x * x * x - x - 1.0

    var r = root_bisect[fn_](1.0, 2.0)
    assert r.success and tol(r.root, 1.3247179572447458, 1e-8)
    print("test_bisect_poly: PASSED")


def test_bisect_domain_error() raises:
    # f(a) and f(b) same sign — should fail with errno != 0
    def fn_(x: Float64) capturing -> Float64:
        return x * x + 1.0

    var r = root_bisect[fn_](0.0, 2.0)
    assert r.errno != 0 and not r.success
    print("test_bisect_domain_error: PASSED")


# ===----------------------------------------------------------------------=== #
# root_brent
# ===----------------------------------------------------------------------=== #


def test_brent_sin() raises:
    def fn_(x: Float64) capturing -> Float64:
        return sin(x)

    var r = root_brent[fn_](3.0, 4.0)
    assert r.success and tol(r.root, 3.141592653589793, 1e-10)
    print("test_brent_sin: PASSED")


def test_brent_exp() raises:
    # exp(x) - 2 = 0, root = ln(2)
    def fn_(x: Float64) capturing -> Float64:
        return exp(x) - 2.0

    var r = root_brent[fn_](0.0, 1.0)
    assert r.success and tol(r.root, 0.6931471805599453, 1e-10)
    print("test_brent_exp: PASSED")


def test_brent_fewer_evals_than_bisect() raises:
    def fn_(x: Float64) capturing -> Float64:
        return x * x * x - x - 1.0

    var rb = root_bisect[fn_](1.0, 2.0)
    var rr = root_brent[fn_](1.0, 2.0)
    assert rr.success and rr.nfev <= rb.nfev
    print("test_brent_fewer_evals_than_bisect: PASSED")


def test_brent_domain_error() raises:
    def fn_(x: Float64) capturing -> Float64:
        return x * x + 1.0

    var r = root_brent[fn_](0.0, 2.0)
    assert r.errno != 0 and not r.success
    print("test_brent_domain_error: PASSED")


# ===----------------------------------------------------------------------=== #
# root_newton
# ===----------------------------------------------------------------------=== #


def test_newton_sqrt2() raises:
    # x^2 - 2 = 0, root = sqrt(2)
    def fn_(x: Float64) capturing -> Float64:
        return x * x - 2.0

    def dfn_(x: Float64) capturing -> Float64:
        return 2.0 * x

    var r = root_newton[fn_, dfn_](1.5)
    assert r.success and tol(r.root, 1.4142135623730951, 1e-10)
    assert r.nfev < 30
    print("test_newton_sqrt2: PASSED")


def test_newton_sin() raises:
    def fn_(x: Float64) capturing -> Float64:
        return sin(x)

    def dfn_(x: Float64) capturing -> Float64:
        return cos(x)

    var r = root_newton[fn_, dfn_](3.0)
    assert r.success and tol(r.root, 3.141592653589793, 1e-10)
    print("test_newton_sin: PASSED")


# ===----------------------------------------------------------------------=== #
# root_secant
# ===----------------------------------------------------------------------=== #


def test_secant_sqrt2() raises:
    def fn_(x: Float64) capturing -> Float64:
        return x * x - 2.0

    var r = root_secant[fn_](1.0, 2.0)
    assert r.success and tol(r.root, 1.4142135623730951, 1e-8)
    print("test_secant_sqrt2: PASSED")


def test_secant_sin() raises:
    def fn_(x: Float64) capturing -> Float64:
        return sin(x)

    var r = root_secant[fn_](3.0, 3.5)
    assert r.success and tol(r.root, 3.141592653589793, 1e-8)
    print("test_secant_sin: PASSED")


# ===----------------------------------------------------------------------=== #
# min_brent
# ===----------------------------------------------------------------------=== #


def test_brent_min_parabola() raises:
    # f(x) = (x-2)^2, min at x=2
    def fn_(x: Float64) capturing -> Float64:
        return (x - 2.0) * (x - 2.0)

    var r = min_brent[fn_](0.0, 1.0, 4.0)
    assert r.success and tol(r.x, 2.0, 1e-6) and tol(r.fun, 0.0, 1e-10)
    print("test_brent_min_parabola: PASSED")


def test_brent_min_cos() raises:
    # f(x) = -cos(x), min at x=0 (in [-1, 1])
    def fn_(x: Float64) capturing -> Float64:
        return -cos(x)

    var r = min_brent[fn_](-1.0, 0.5, 1.0)
    assert r.success and tol(r.x, 0.0, 1e-6) and tol(r.fun, -1.0, 1e-10)
    print("test_brent_min_cos: PASSED")


def test_brent_min_captures_closure() raises:
    var shift: Float64 = 3.0

    def fn_(x: Float64) capturing -> Float64:
        return (x - shift) * (x - shift)

    var r = min_brent[fn_](1.0, 2.0, 5.0)
    assert r.success and tol(r.x, 3.0, 1e-6)
    print("test_brent_min_captures_closure: PASSED")


# ===----------------------------------------------------------------------=== #
# min_golden
# ===----------------------------------------------------------------------=== #


def test_golden_parabola() raises:
    def fn_(x: Float64) capturing -> Float64:
        return (x - 1.5) * (x - 1.5)

    var r = min_golden[fn_](0.0, 0.5, 3.0)
    assert r.success and tol(r.x, 1.5, 1e-6)
    print("test_golden_parabola: PASSED")


def test_golden_cos() raises:
    def fn_(x: Float64) capturing -> Float64:
        return -cos(x)

    var r = min_golden[fn_](-1.0, 0.5, 1.0)
    assert r.success and tol(r.x, 0.0, 1e-6)
    print("test_golden_cos: PASSED")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("All optimizer tests PASSED")
