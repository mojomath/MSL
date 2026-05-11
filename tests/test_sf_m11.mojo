from std.testing import TestSuite
from std.math import abs, exp, log

from msl.sf.psi import psi, psi_n
from msl.sf.gamma_inc import gamma_inc_P, gamma_inc_Q, gamma_inc
from msl.sf.beta_inc import beta_inc


def tolerance(x: Float64, expected: Float64, tol: Float64) -> Bool:
    return abs(x - expected) < tol


# ===----------------------------------------------------------------------=== #
# psi (digamma) tests
# ===----------------------------------------------------------------------=== #


def test_psi_integer_1() raises:
    # psi(1) = -EULER
    var r = psi(1.0)
    assert r.errno == 0
    assert tolerance(r.val, -0.5772156649015329, 1e-10)
    print("test_psi_integer_1: PASSED")


def test_psi_integer_2() raises:
    # psi(2) = 1 - EULER
    var r = psi(2.0)
    assert r.errno == 0
    assert tolerance(r.val, 0.4227843350984671, 1e-10)
    print("test_psi_integer_2: PASSED")


def test_psi_integer_4() raises:
    # psi(4) = 1 + 1/2 + 1/3 - EULER
    var r = psi(4.0)
    assert r.errno == 0
    assert tolerance(r.val, 1.2561176684318004, 1e-10)
    print("test_psi_integer_4: PASSED")


def test_psi_half() raises:
    # psi(1/2) = -EULER - 2*ln(2)
    var r = psi(0.5)
    assert r.errno == 0
    assert tolerance(r.val, -1.9635100260214235, 1e-9)
    print("test_psi_half: PASSED")


def test_psi_large() raises:
    # psi(100) ~ log(100) - 1/200
    var r = psi(100.0)
    assert r.errno == 0
    assert tolerance(r.val, 4.600161852738087, 1e-9)
    print("test_psi_large: PASSED")


def test_psi_domain_error() raises:
    var r = psi(-1.0)
    assert r.errno != 0
    print("test_psi_domain_error: PASSED")


# ===----------------------------------------------------------------------=== #
# gamma_inc_P / gamma_inc_Q tests
# ===----------------------------------------------------------------------=== #


def test_gamma_inc_P_basic() raises:
    # P(1, x) = 1 - exp(-x)
    var r = gamma_inc_P(1.0, 1.0)
    assert r.errno == 0
    assert tolerance(r.val, 1.0 - exp(-1.0), 1e-10)
    print("test_gamma_inc_P_basic: PASSED")


def test_gamma_inc_Q_basic() raises:
    # Q(1, x) = exp(-x)
    var r = gamma_inc_Q(1.0, 1.0)
    assert r.errno == 0
    assert tolerance(r.val, exp(-1.0), 1e-10)
    print("test_gamma_inc_Q_basic: PASSED")


def test_gamma_inc_PQ_sum_to_one() raises:
    var p = gamma_inc_P(2.5, 3.0)
    var q = gamma_inc_Q(2.5, 3.0)
    assert p.errno == 0 and q.errno == 0
    assert tolerance(p.val + q.val, 1.0, 1e-12)
    print("test_gamma_inc_PQ_sum_to_one: PASSED")


def test_gamma_inc_P_zero_x() raises:
    var r = gamma_inc_P(2.0, 0.0)
    assert r.errno == 0
    assert r.val == 0.0
    print("test_gamma_inc_P_zero_x: PASSED")


def test_gamma_inc_Q_zero_x() raises:
    var r = gamma_inc_Q(2.0, 0.0)
    assert r.errno == 0
    assert r.val == 1.0
    print("test_gamma_inc_Q_zero_x: PASSED")


def test_gamma_inc_P_large_x() raises:
    # P(a, x) -> 1 as x -> inf
    var r = gamma_inc_P(2.0, 100.0)
    assert r.errno == 0
    assert tolerance(r.val, 1.0, 1e-10)
    print("test_gamma_inc_P_large_x: PASSED")


def test_gamma_inc_domain_error() raises:
    var r = gamma_inc_P(-1.0, 1.0)
    assert r.errno != 0
    print("test_gamma_inc_domain_error: PASSED")


# ===----------------------------------------------------------------------=== #
# beta_inc tests
# ===----------------------------------------------------------------------=== #


def test_beta_inc_half() raises:
    # I_{0.5}(a,a) = 0.5 by symmetry for any a
    var r = beta_inc(2.0, 2.0, 0.5)
    assert r.errno == 0
    assert tolerance(r.val, 0.5, 1e-10)
    print("test_beta_inc_half: PASSED")


def test_beta_inc_boundary_0() raises:
    var r = beta_inc(2.0, 3.0, 0.0)
    assert r.errno == 0
    assert r.val == 0.0
    print("test_beta_inc_boundary_0: PASSED")


def test_beta_inc_boundary_1() raises:
    var r = beta_inc(2.0, 3.0, 1.0)
    assert r.errno == 0
    assert r.val == 1.0
    print("test_beta_inc_boundary_1: PASSED")


def test_beta_inc_known_value() raises:
    # I_{0.25}(2,3) = 0.26171875 (from tables)
    var r = beta_inc(2.0, 3.0, 0.25)
    assert r.errno == 0
    assert tolerance(r.val, 0.26171875, 1e-8)
    print("test_beta_inc_known_value: PASSED")


def test_beta_inc_symmetry() raises:
    # I_x(a,b) + I_{1-x}(b,a) = 1
    var x: Float64 = 0.3
    var p = beta_inc(2.0, 5.0, x)
    var q = beta_inc(5.0, 2.0, 1.0 - x)
    assert p.errno == 0 and q.errno == 0
    assert tolerance(p.val + q.val, 1.0, 1e-12)
    print("test_beta_inc_symmetry: PASSED")


def test_beta_inc_domain_error() raises:
    var r = beta_inc(-1.0, 2.0, 0.5)
    assert r.errno != 0
    print("test_beta_inc_domain_error: PASSED")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("All M11 (psi, gamma_inc, beta_inc) tests PASSED")
