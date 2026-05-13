from std.testing import TestSuite
from std.math import abs, sin, cos, exp, sqrt, log

from msl.deriv import deriv_central, deriv_forward, deriv_backward


def tolerance(x: Float64, expected: Float64, tol: Float64) -> Bool:
    return abs(x - expected) < tol


# ===----------------------------------------------------------------------=== #
# Central differences
# ===----------------------------------------------------------------------=== #


def test_central_polynomial() raises:
    # f(x) = x^3, f'(x) = 3x^2, f'(2) = 12
    def fn_(x: Float64) capturing -> Float64:
        return x * x * x

    var r = deriv_central[fn_](2.0)
    assert tolerance(r.val, 12.0, 1e-8)
    assert r.err < 1e-5
    print("test_central_polynomial: PASSED")


def test_central_sin() raises:
    # f'(x) = cos(x), f'(pi/4) = cos(pi/4) = sqrt(2)/2
    def fn_(x: Float64) capturing -> Float64:
        return sin(x)

    var r = deriv_central[fn_](0.7853981633974483)
    assert tolerance(r.val, cos(0.7853981633974483), 1e-10)
    print("test_central_sin: PASSED")


def test_central_exp() raises:
    # f'(x) = exp(x), f'(1) = e
    def fn_(x: Float64) capturing -> Float64:
        return exp(x)

    var r = deriv_central[fn_](1.0)
    assert tolerance(r.val, 2.718281828459045, 1e-10)
    print("test_central_exp: PASSED")


def test_central_log() raises:
    # f'(x) = 1/x, f'(2) = 0.5
    def fn_(x: Float64) capturing -> Float64:
        return log(x)

    var r = deriv_central[fn_](2.0)
    assert tolerance(r.val, 0.5, 1e-10)
    print("test_central_log: PASSED")


def test_central_at_zero() raises:
    # f(x) = x^2, f'(0) = 0
    def fn_(x: Float64) capturing -> Float64:
        return x * x

    var r = deriv_central[fn_](0.0)
    assert tolerance(r.val, 0.0, 1e-8)
    print("test_central_at_zero: PASSED")


def test_central_explicit_step() raises:
    # Manual h should give same answer as auto
    def fn_(x: Float64) capturing -> Float64:
        return x * x * x

    var r = deriv_central[fn_](2.0, h=1e-4)
    assert tolerance(r.val, 12.0, 1e-8)
    print("test_central_explicit_step: PASSED")


def test_central_error_bound() raises:
    # err should be small for smooth function
    def fn_(x: Float64) capturing -> Float64:
        return sin(x)

    var r = deriv_central[fn_](1.0)
    assert r.err < 1e-8
    print("test_central_error_bound: PASSED")


def test_central_captures_closure() raises:
    var a: Float64 = 3.0

    def fn_(x: Float64) capturing -> Float64:
        return x ** a

    # f'(x) = 3*x^2, f'(2) = 12
    var r = deriv_central[fn_](2.0)
    assert tolerance(r.val, 12.0, 1e-8)
    print("test_central_captures_closure: PASSED")


# ===----------------------------------------------------------------------=== #
# Forward differences
# ===----------------------------------------------------------------------=== #


def test_forward_polynomial() raises:
    def fn_(x: Float64) capturing -> Float64:
        return x * x * x

    var r = deriv_forward[fn_](2.0)
    assert tolerance(r.val, 12.0, 1e-6)
    print("test_forward_polynomial: PASSED")


def test_forward_sin() raises:
    def fn_(x: Float64) capturing -> Float64:
        return sin(x)

    var r = deriv_forward[fn_](0.0)
    assert tolerance(r.val, 1.0, 1e-8)
    print("test_forward_sin: PASSED")


def test_forward_explicit_step() raises:
    def fn_(x: Float64) capturing -> Float64:
        return x * x

    var r = deriv_forward[fn_](1.0, h=1e-4)
    assert tolerance(r.val, 2.0, 1e-6)
    print("test_forward_explicit_step: PASSED")


def test_forward_error_bound() raises:
    def fn_(x: Float64) capturing -> Float64:
        return x * x * x

    var r = deriv_forward[fn_](2.0)
    assert r.err < 1e-4
    print("test_forward_error_bound: PASSED")


# ===----------------------------------------------------------------------=== #
# Backward differences
# ===----------------------------------------------------------------------=== #


def test_backward_polynomial() raises:
    def fn_(x: Float64) capturing -> Float64:
        return x * x * x

    var r = deriv_backward[fn_](2.0)
    assert tolerance(r.val, 12.0, 1e-6)
    print("test_backward_polynomial: PASSED")


def test_backward_sin() raises:
    def fn_(x: Float64) capturing -> Float64:
        return sin(x)

    var r = deriv_backward[fn_](3.141592653589793)
    assert tolerance(r.val, cos(3.141592653589793), 1e-8)
    print("test_backward_sin: PASSED")


def test_backward_error_bound() raises:
    def fn_(x: Float64) capturing -> Float64:
        return x * x * x

    var r = deriv_backward[fn_](2.0)
    assert r.err < 1e-4
    print("test_backward_error_bound: PASSED")


def test_central_better_than_forward() raises:
    # Central should give smaller error for same step
    def fn_(x: Float64) capturing -> Float64:
        return sin(x)

    var rc = deriv_central[fn_](1.0)
    var rf = deriv_forward[fn_](1.0)
    assert rc.err <= rf.err + 1e-15
    print("test_central_better_than_forward: PASSED")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("All deriv tests PASSED")
