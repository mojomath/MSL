from std.testing import TestSuite
from std.math import abs, exp, log, sqrt

from msl.rng import RNG, MT19937
from msl.distributions import (
    tdist, tdist_pdf,
    lognormal, lognormal_pdf,
    weibull, weibull_pdf,
    binomial, binomial_pdf,
    negative_binomial, negative_binomial_pdf,
    cauchy, cauchy_pdf,
    laplace, laplace_pdf,
)


def tolerance(x: Float64, expected: Float64, tol: Float64) -> Bool:
    return abs(x - expected) < tol


# ===----------------------------------------------------------------------=== #
# Student's t
# ===----------------------------------------------------------------------=== #


def test_tdist_sample_range() raises:
    var rng = RNG[MT19937](MT19937(42), 42)
    for _ in range(100):
        var v = tdist(rng, 5.0)
        assert v > -50.0 and v < 50.0
    print("test_tdist_sample_range: PASSED")


def test_tdist_pdf_symmetric() raises:
    assert tolerance(tdist_pdf(1.0, 5.0), tdist_pdf(-1.0, 5.0), 1e-15)
    print("test_tdist_pdf_symmetric: PASSED")


def test_tdist_pdf_peak() raises:
    # PDF at 0 should be maximum
    assert tdist_pdf(0.0, 5.0) > tdist_pdf(1.0, 5.0)
    print("test_tdist_pdf_peak: PASSED")


def test_tdist_large_nu() raises:
    # For large nu, t-dist approaches normal
    var rng = RNG[MT19937](MT19937(99), 99)
    var mean: Float64 = 0.0
    for _ in range(5000):
        mean += tdist(rng, 100.0)
    mean /= 5000.0
    assert abs(mean) < 0.1
    print("test_tdist_large_nu: PASSED")


# ===----------------------------------------------------------------------=== #
# Log-normal
# ===----------------------------------------------------------------------=== #


def test_lognormal_positive() raises:
    var rng = RNG[MT19937](MT19937(1), 1)
    for _ in range(100):
        assert lognormal(rng, 0.0, 1.0) > 0.0
    print("test_lognormal_positive: PASSED")


def test_lognormal_pdf_zero() raises:
    assert lognormal_pdf(0.0, 0.0, 1.0) == 0.0
    assert lognormal_pdf(-1.0, 0.0, 1.0) == 0.0
    print("test_lognormal_pdf_zero: PASSED")


def test_lognormal_pdf_value() raises:
    # At x=1, zeta=0, sigma=1: pdf = 1/sqrt(2pi) ≈ 0.3989
    assert tolerance(lognormal_pdf(1.0, 0.0, 1.0), 0.3989422804014327, 1e-8)
    print("test_lognormal_pdf_value: PASSED")


# ===----------------------------------------------------------------------=== #
# Weibull
# ===----------------------------------------------------------------------=== #


def test_weibull_positive() raises:
    var rng = RNG[MT19937](MT19937(2), 2)
    for _ in range(100):
        assert weibull(rng, 1.0, 2.0) > 0.0
    print("test_weibull_positive: PASSED")


def test_weibull_pdf_zero_x() raises:
    # b=1 (exponential): pdf(0) = 1/a
    assert tolerance(weibull_pdf(0.0, 2.0, 1.0), 0.5, 1e-15)
    # b>1: pdf(0) = 0
    assert weibull_pdf(0.0, 1.0, 2.0) == 0.0
    print("test_weibull_pdf_zero_x: PASSED")


def test_weibull_mean() raises:
    # Weibull(a=1, b=1) = Exponential(1), mean=1
    var rng = RNG[MT19937](MT19937(3), 3)
    var mean: Float64 = 0.0
    for _ in range(10000):
        mean += weibull(rng, 1.0, 1.0)
    mean /= 10000.0
    assert tolerance(mean, 1.0, 0.05)
    print("test_weibull_mean: PASSED")


# ===----------------------------------------------------------------------=== #
# Binomial
# ===----------------------------------------------------------------------=== #


def test_binomial_range() raises:
    var rng = RNG[MT19937](MT19937(4), 4)
    for _ in range(100):
        var k = binomial(rng, 0.5, 20)
        assert k >= 0 and k <= 20
    print("test_binomial_range: PASSED")


def test_binomial_p0() raises:
    var rng = RNG[MT19937](MT19937(5), 5)
    assert binomial(rng, 0.0, 10) == 0
    print("test_binomial_p0: PASSED")


def test_binomial_p1() raises:
    var rng = RNG[MT19937](MT19937(6), 6)
    assert binomial(rng, 1.0, 10) == 10
    print("test_binomial_p1: PASSED")


def test_binomial_pdf_sum() raises:
    # PMF must sum to 1
    var total: Float64 = 0.0
    for k in range(11):
        total += binomial_pdf(k, 0.3, 10)
    assert tolerance(total, 1.0, 1e-10)
    print("test_binomial_pdf_sum: PASSED")


def test_binomial_mean() raises:
    var rng = RNG[MT19937](MT19937(7), 7)
    var mean: Float64 = 0.0
    for _ in range(5000):
        mean += Float64(binomial(rng, 0.4, 20))
    mean /= 5000.0
    # E[X] = n*p = 8
    assert tolerance(mean, 8.0, 0.2)
    print("test_binomial_mean: PASSED")


# ===----------------------------------------------------------------------=== #
# Negative Binomial
# ===----------------------------------------------------------------------=== #


def test_negative_binomial_nonneg() raises:
    var rng = RNG[MT19937](MT19937(8), 8)
    for _ in range(100):
        assert negative_binomial(rng, 0.5, 3.0) >= 0
    print("test_negative_binomial_nonneg: PASSED")


def test_negative_binomial_pdf_positive() raises:
    var p = negative_binomial_pdf(2, 0.5, 3.0)
    assert p > 0.0 and p < 1.0
    print("test_negative_binomial_pdf_positive: PASSED")


# ===----------------------------------------------------------------------=== #
# Cauchy
# ===----------------------------------------------------------------------=== #


def test_cauchy_sample() raises:
    var rng = RNG[MT19937](MT19937(9), 9)
    var count_near_zero: Int = 0
    for _ in range(1000):
        var v = cauchy(rng, 1.0)
        if abs(v) < 2.0:
            count_near_zero += 1
    # Most samples should be near median (heavy-tailed but concentrated around 0)
    assert count_near_zero > 400
    print("test_cauchy_sample: PASSED")


def test_cauchy_pdf_peak() raises:
    # PDF at 0 = 1/(pi*a)
    assert tolerance(cauchy_pdf(0.0, 1.0), 1.0 / 3.141592653589793, 1e-10)
    print("test_cauchy_pdf_peak: PASSED")


def test_cauchy_pdf_symmetric() raises:
    assert tolerance(cauchy_pdf(2.0, 1.0), cauchy_pdf(-2.0, 1.0), 1e-15)
    print("test_cauchy_pdf_symmetric: PASSED")


# ===----------------------------------------------------------------------=== #
# Laplace
# ===----------------------------------------------------------------------=== #


def test_laplace_mean() raises:
    var rng = RNG[MT19937](MT19937(10), 10)
    var mean: Float64 = 0.0
    for _ in range(10000):
        mean += laplace(rng, 1.0)
    mean /= 10000.0
    assert abs(mean) < 0.05
    print("test_laplace_mean: PASSED")


def test_laplace_pdf_peak() raises:
    # PDF at 0 = 1/(2a)
    assert tolerance(laplace_pdf(0.0, 2.0), 0.25, 1e-15)
    print("test_laplace_pdf_peak: PASSED")


def test_laplace_pdf_symmetric() raises:
    assert tolerance(laplace_pdf(1.0, 1.0), laplace_pdf(-1.0, 1.0), 1e-15)
    print("test_laplace_pdf_symmetric: PASSED")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("All M10 distribution tests PASSED")
