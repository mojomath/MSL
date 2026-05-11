from std.testing import TestSuite
from std.math import abs

from msl.rng import RNG, MT19937
from msl.distributions import gaussian, uniform, exponential, gamma, beta, poisson, gaussian_pdf, uniform_pdf, exponential_pdf, gamma_pdf


def tolerance(x: Float64, expected: Float64, tol: Float64) -> Bool:
    return abs(x - expected) < tol


def test_gaussian() raises:
    var rng = RNG[MT19937](MT19937(42), 42)

    var val = gaussian(rng, 1.0)
    assert val > -5.0 and val < 5.0

    var mean: Float64 = 0.0
    var n = 10000
    for _ in range(n):
        mean += gaussian(rng, 2.0)
    mean /= Float64(n)
    assert tolerance(mean, 0.0, 0.1)

    var val_sigma3 = gaussian(rng, 1.0)
    var pdf = gaussian_pdf(val_sigma3, 1.0)
    assert pdf >= 0.0 and pdf <= 1.0

    var pdf_at_0 = gaussian_pdf(0.0, 1.0)
    assert tolerance(pdf_at_0, 0.3989422804014327, 1e-10)

    var pdf_at_1 = gaussian_pdf(1.0, 1.0)
    assert tolerance(pdf_at_1, 0.24197072473114514, 1e-10)

    print("test_gaussian: PASSED")


def test_uniform() raises:
    var rng = RNG[MT19937](MT19937(123), 123)

    var val = uniform(rng, 0.0, 10.0)
    assert val >= 0.0
    assert val < 10.0

    var mean: Float64 = 0.0
    for _ in range(10000):
        mean += uniform(rng, 5.0, 15.0)
    mean /= 10000.0
    assert tolerance(mean, 10.0, 0.1)

    assert uniform_pdf(-1.0, 0.0, 10.0) == 0.0
    assert uniform_pdf(11.0, 0.0, 10.0) == 0.0
    assert tolerance(uniform_pdf(5.0, 0.0, 10.0), 0.1, 1e-10)
    assert tolerance(uniform_pdf(0.0, 0.0, 10.0), 0.1, 1e-10)
    assert tolerance(uniform_pdf(10.0, 0.0, 10.0), 0.1, 1e-10)

    print("test_uniform: PASSED")


def test_exponential() raises:
    var rng = RNG[MT19937](MT19937(456), 456)

    var val = exponential(rng, 2.0)
    assert val >= 0.0

    var mean: Float64 = 0.0
    for _ in range(10000):
        mean += exponential(rng, 5.0)
    mean /= 10000.0
    assert tolerance(mean, 5.0, 0.2)

    var pdf_at_0 = exponential_pdf(0.0, 2.0)
    assert tolerance(pdf_at_0, 0.5, 1e-10)

    var pdf_at_2 = exponential_pdf(2.0, 2.0)
    assert tolerance(pdf_at_2, 0.36787944117144233, 1e-10)

    assert exponential_pdf(-1.0, 1.0) == 0.0

    print("test_exponential: PASSED")


def test_gamma() raises:
    var rng = RNG[MT19937](MT19937(789), 789)

    var val = gamma(rng, 2.0, 1.0)
    assert val >= 0.0

    var mean: Float64 = 0.0
    for _ in range(10000):
        mean += gamma(rng, 3.0, 2.0)
    mean /= 10000.0
    assert tolerance(mean, 6.0, 0.3)

    var pdf_at_1 = gamma_pdf(1.0, 2.0, 1.0)
    assert pdf_at_1 >= 0.0

    assert gamma_pdf(0.0, 2.0, 1.0) == 0.0
    assert gamma_pdf(-1.0, 2.0, 1.0) == 0.0

    print("test_gamma: PASSED")


def test_beta() raises:
    var rng = RNG[MT19937](MT19937(111), 111)

    var val = beta(rng, 2.0, 3.0)
    assert val >= 0.0 and val <= 1.0

    var mean: Float64 = 0.0
    for _ in range(5000):
        mean += beta(rng, 3.0, 5.0)
    mean /= 5000.0
    assert tolerance(mean, 3.0 / 8.0, 0.05)

    var all_valid = True
    for _ in range(100):
        var v = beta(rng, 0.5, 0.5)
        if v < 0.0 or v > 1.0:
            all_valid = False
    assert all_valid

    print("test_beta: PASSED")


def test_poisson() raises:
    var rng = RNG[MT19937](MT19937(222), 222)

    var val = poisson(rng, 5.0)
    assert val >= 0

    var mean: Float64 = 0.0
    for _ in range(5000):
        mean += Float64(poisson(rng, 10.0))
    mean /= 5000.0
    assert tolerance(mean, 10.0, 0.5)

    var variance: Float64 = 0.0
    for _ in range(5000):
        var v = Float64(poisson(rng, 10.0))
        variance += (v - mean) * (v - mean)
    variance /= 5000.0
    assert tolerance(variance, 10.0, 1.0)

    print("test_poisson: PASSED")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("All dist tests PASSED")
