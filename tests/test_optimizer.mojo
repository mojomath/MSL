from std.testing import TestSuite
from std.math import abs, sin, cos, sqrt, exp

from msl.optimizer import (
    root_bisect,
    root_brent,
    root_newton,
    root_secant,
    root_falsepos,
    root_steffenson,
)
from msl.optimizer import min_brent, min_golden
from msl.optimizer import (
    root_test_interval,
    root_test_residual,
    root_test_delta,
    min_test_interval,
    min_find_bracket,
)
from msl.core.errno import MSL_SUCCESS, MSL_CONTINUE, MSL_EBADTOL, MSL_EINVAL


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
    # f(a) and f(b) same sign - should fail with errno != 0
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
# root_falsepos
# ===----------------------------------------------------------------------=== #


def test_falsepos_sin() raises:
    def fn_(x: Float64) capturing -> Float64:
        return sin(x)

    var r = root_falsepos[fn_](3.0, 4.0)
    assert r.success and tol(r.root, 3.141592653589793, 1e-8)
    print("test_falsepos_sin: PASSED")


def test_falsepos_poly() raises:
    def fn_(x: Float64) capturing -> Float64:
        return x * x * x - x - 1.0

    var r = root_falsepos[fn_](1.0, 2.0)
    assert r.success and tol(r.root, 1.3247179572447458, 1e-8)
    print("test_falsepos_poly: PASSED")


def test_falsepos_domain_error() raises:
    def fn_(x: Float64) capturing -> Float64:
        return x * x + 1.0

    var r = root_falsepos[fn_](0.0, 2.0)
    assert r.errno != 0 and not r.success
    print("test_falsepos_domain_error: PASSED")


# ===----------------------------------------------------------------------=== #
# root_steffenson
# ===----------------------------------------------------------------------=== #


def test_steffenson_sqrt2() raises:
    def fn_(x: Float64) capturing -> Float64:
        return x * x - 2.0

    def dfn_(x: Float64) capturing -> Float64:
        return 2.0 * x

    var r = root_steffenson[fn_, dfn_](1.5)
    assert r.success and tol(r.root, 1.4142135623730951, 1e-10)
    print("test_steffenson_sqrt2: PASSED")


def test_steffenson_multiple_root() raises:
    # Multiple root at x=1 where plain Newton is only linearly convergent.
    def fn_(x: Float64) capturing -> Float64:
        var y = x - 1.0
        return y * y

    def dfn_(x: Float64) capturing -> Float64:
        return 2.0 * (x - 1.0)

    var r = root_steffenson[fn_, dfn_](2.0, max_iter=200)
    assert r.success and tol(r.root, 1.0, 1e-8)
    print("test_steffenson_multiple_root: PASSED")


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


# ===----------------------------------------------------------------------=== #
# convergence/bracketing helpers
# ===----------------------------------------------------------------------=== #


def test_root_test_interval_success() raises:
    var status = root_test_interval(1.0, 1.0 + 1e-12, 1e-10, 1e-10)
    assert status == MSL_SUCCESS
    print("test_root_test_interval_success: PASSED")


def test_root_test_interval_continue() raises:
    var status = root_test_interval(1.0, 2.0, 1e-12, 1e-12)
    assert status == MSL_CONTINUE
    print("test_root_test_interval_continue: PASSED")


def test_root_test_interval_invalid_tolerance() raises:
    var status = root_test_interval(0.0, 1.0, -1e-10, 1e-10)
    assert status == MSL_EBADTOL
    print("test_root_test_interval_invalid_tolerance: PASSED")


def test_root_test_interval_invalid_bounds() raises:
    var status = root_test_interval(2.0, 1.0, 1e-10, 1e-10)
    assert status == MSL_EINVAL
    print("test_root_test_interval_invalid_bounds: PASSED")


def test_root_test_delta_and_residual() raises:
    assert root_test_delta(1.0, 1.0 + 1e-13, 1e-10, 1e-10) == MSL_SUCCESS
    assert root_test_delta(1.0, 1.1, 1e-14, 1e-14) == MSL_CONTINUE
    assert root_test_residual(1e-12, 1e-10) == MSL_SUCCESS
    assert root_test_residual(1e-4, 1e-10) == MSL_CONTINUE
    print("test_root_test_delta_and_residual: PASSED")


def test_min_test_interval() raises:
    assert min_test_interval(0.0, 0.0 + 1e-12, 1e-10, 1e-10) == MSL_SUCCESS
    assert min_test_interval(-1.0, 1.0, 1e-14, 1e-14) == MSL_CONTINUE
    assert min_test_interval(0.0, 1.0, -1e-10, 1e-10) == MSL_EBADTOL
    print("test_min_test_interval: PASSED")


def test_min_find_bracket_parabola() raises:
    def fn_(x: Float64) capturing -> Float64:
        return (x - 2.0) * (x - 2.0)

    var br = min_find_bracket[fn_](0.0, 1.0, eval_max=200)
    assert br.success
    assert br.x_lower < br.x_minimum and br.x_minimum < br.x_upper
    assert br.f_minimum <= br.f_lower and br.f_minimum <= br.f_upper
    print("test_min_find_bracket_parabola: PASSED")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("All optimizer tests PASSED")
