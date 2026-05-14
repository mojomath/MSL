from std.testing import TestSuite
from std.math import abs, sin, cos, exp, sqrt, log

from msl.integration import qag, qags, MSL_INTEG_GAUSS15, MSL_INTEG_GAUSS21, MSL_INTEG_GAUSS61


def tolerance(x: Float64, expected: Float64, tol: Float64) -> Bool:
    return abs(x - expected) < tol


# ===----------------------------------------------------------------------=== #
# QAG tests
# ===----------------------------------------------------------------------=== #


def test_qag_linear() raises:
    def integrand(x: Float64) capturing -> Float64:
        return x

    var r = qag[integrand](0.0, 1.0, 0.0, 1e-10)
    assert tolerance(r.val, 0.5, 1e-10)
    assert r.err < 1e-8
    print("test_qag_linear: PASSED")


def test_qag_sin() raises:
    def integrand(x: Float64) capturing -> Float64:
        return sin(x)

    var r = qag[integrand](0.0, 3.141592653589793, 0.0, 1e-10)
    assert tolerance(r.val, 2.0, 1e-10)
    assert r.err < 1e-8
    print("test_qag_sin: PASSED")


def test_qag_exp() raises:
    def integrand(x: Float64) capturing -> Float64:
        return exp(x)

    var r = qag[integrand](0.0, 1.0, 0.0, 1e-10)
    assert tolerance(r.val, 1.7182818284590452, 1e-10)
    assert r.err < 1e-8
    print("test_qag_exp: PASSED")


def test_qag_oscillatory() raises:
    # integral of sin(100*x) over [0, pi] ≈ 0.02
    def integrand(x: Float64) capturing -> Float64:
        return sin(100.0 * x)

    var r = qag[integrand](0.0, 3.141592653589793, 1e-8, 1e-8, limit=200, key=MSL_INTEG_GAUSS61)
    assert r.err < 1e-5
    print("test_qag_oscillatory: PASSED")


def test_qag_rule_selection() raises:
    def integrand(x: Float64) capturing -> Float64:
        return x * x * x

    var r15 = qag[integrand](0.0, 1.0, 0.0, 1e-12, key=MSL_INTEG_GAUSS15)
    var r21 = qag[integrand](0.0, 1.0, 0.0, 1e-12, key=MSL_INTEG_GAUSS21)
    assert tolerance(r15.val, 0.25, 1e-12)
    assert tolerance(r21.val, 0.25, 1e-12)
    print("test_qag_rule_selection: PASSED")


def test_qag_captures_closure() raises:
    var alpha: Float64 = 3.0

    def integrand(x: Float64) capturing -> Float64:
        return x ** alpha

    # integral of x^3 = 0.25
    var r = qag[integrand](0.0, 1.0, 0.0, 1e-10)
    assert tolerance(r.val, 0.25, 1e-10)
    print("test_qag_captures_closure: PASSED")


# ===----------------------------------------------------------------------=== #
# QAGS tests
# ===----------------------------------------------------------------------=== #


def test_qags_linear() raises:
    def integrand(x: Float64) capturing -> Float64:
        return x

    var r = qags[integrand](0.0, 1.0, 0.0, 1e-10)
    assert tolerance(r.val, 0.5, 1e-10)
    assert r.err < 1e-8
    print("test_qags_linear: PASSED")


def test_qags_sin() raises:
    def integrand(x: Float64) capturing -> Float64:
        return sin(x)

    var r = qags[integrand](0.0, 3.141592653589793, 0.0, 1e-10)
    assert tolerance(r.val, 2.0, 1e-10)
    assert r.err < 1e-8
    print("test_qags_sin: PASSED")


def test_qags_exp() raises:
    def integrand(x: Float64) capturing -> Float64:
        return exp(x)

    var r = qags[integrand](0.0, 1.0, 0.0, 1e-10)
    assert tolerance(r.val, 1.7182818284590452, 1e-10)
    assert r.err < 1e-8
    print("test_qags_exp: PASSED")


def test_qags_log_singularity() raises:
    # integral of log(x) over [0,1] = -1, integrand has singularity at 0
    def integrand(x: Float64) capturing -> Float64:
        return log(x)

    var r = qags[integrand](0.0001, 1.0, 1e-8, 1e-8)
    # approximate: integral from 0.0001 to 1 of log(x) ≈ -1 + 0.0001*(log(0.0001)-1)
    var expected = -1.0 + 0.0001 * (log(0.0001) - 1.0)
    assert tolerance(r.val, expected, 1e-6)
    print("test_qags_log_singularity: PASSED")


def test_qags_sqrt_singularity() raises:
    # integral of 1/sqrt(x) over [0,1] = 2
    def integrand(x: Float64) capturing -> Float64:
        return 1.0 / sqrt(x)

    var r = qags[integrand](1e-8, 1.0, 1e-8, 1e-8)
    var expected = 2.0 - 2.0 * sqrt(1e-8)
    assert tolerance(r.val, expected, 1e-5)
    print("test_qags_sqrt_singularity: PASSED")


def test_qags_captures_closure() raises:
    var omega: Float64 = 10.0

    def integrand(x: Float64) capturing -> Float64:
        return sin(omega * x)

    # integral of sin(10x) from 0 to pi = (1 - cos(10*pi)) / 10 = 0
    var r = qags[integrand](0.0, 3.141592653589793, 1e-8, 1e-8)
    assert r.err < 1e-5
    print("test_qags_captures_closure: PASSED")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("All adaptive integration (QAG/QAGS) tests PASSED")
