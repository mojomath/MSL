# SPDX-License-Identifier: GPL-3.0-or-later

# TODO: Add scipy dependancy to cross check these values.

from std.testing import TestSuite
from std.math import abs

from msl.sf import (
    airy_ai,
    airy_bi,
    airy_ai_scaled,
    airy_bi_scaled,
    bessel_j0,
    bessel_j1,
    bessel_y0,
    bessel_y1,
    bessel_i0_scaled,
    bessel_i1_scaled,
    bessel_k0_scaled,
    bessel_k1_scaled,
    gamma,
    lngamma,
    gammastar,
    gammainv,
    factorial,
    double_factorial,
    ln_factorial,
    ln_double_factorial,
    beta,
    lnbeta,
    erf,
    erfc,
    log_erfc,
    erf_Z,
    erf_Q,
    hazard,
    legendre_P1,
    legendre_P2,
    legendre_P3,
    legendre_Pl,
    SFSResult,
)


def tolerance(x: Float64, expected: Float64, tol: Float64) -> Bool:
    return abs(x - expected) < tol


def test_airy_ai_zero() raises:
    var result = airy_ai(0.0)
    var expected = 0.355028053887835
    assert tolerance(result.val, expected, 1e-10)
    print("test_airy_ai_zero: PASSED")


def test_airy_ai_negative() raises:
    var result = airy_ai(-1.0)
    var expected = -0.108310073618049
    assert tolerance(result.val, expected, 1e-10)
    print("test_airy_ai_negative: PASSED")


def test_airy_ai_positive() raises:
    var result = airy_ai(1.0)
    var expected = 0.135887128101646
    assert tolerance(result.val, expected, 1e-10)
    print("test_airy_ai_positive: PASSED")


def test_airy_bi_zero() raises:
    var result = airy_bi(0.0)
    var expected = 0.614926627446005
    assert tolerance(result.val, expected, 1e-10)
    print("test_airy_bi_zero: PASSED")


def test_airy_bi_negative() raises:
    var result = airy_bi(-1.0)
    var expected = -0.669913157408913
    assert tolerance(result.val, expected, 1e-10)
    print("test_airy_bi_negative: PASSED")


def test_airy_bi_positive() raises:
    var result = airy_bi(1.0)
    var expected = 0.972326463456219
    assert tolerance(result.val, expected, 1e-10)
    print("test_airy_bi_positive: PASSED")


def test_airy_ai_scaled() raises:
    var result = airy_ai_scaled(2.0)
    var expected = 0.012126486879842
    assert tolerance(result.val, expected, 1e-10)
    print("test_airy_ai_scaled: PASSED")


def test_airy_bi_scaled() raises:
    var result = airy_bi_scaled(2.0)
    var expected = 1.372449140043837
    assert tolerance(result.val, expected, 1e-10)
    print("test_airy_bi_scaled: PASSED")


def test_bessel_j0() raises:
    var result = bessel_j0(1.0)
    var expected = 0.765197686557967
    assert tolerance(result.val, expected, 1e-10)
    print("test_bessel_j0: PASSED")


def test_bessel_j1() raises:
    var result = bessel_j1(1.0)
    var expected = 0.440050585744933
    assert tolerance(result.val, expected, 1e-10)
    print("test_bessel_j1: PASSED")


def test_bessel_y0() raises:
    var result = bessel_y0(1.0)
    var expected = -0.088256964830592
    assert tolerance(result.val, expected, 1e-10)
    print("test_bessel_y0: PASSED")


def test_bessel_y1() raises:
    var result = bessel_y1(1.0)
    var expected = -0.781212418297592
    assert tolerance(result.val, expected, 1e-10)
    print("test_bessel_y1: PASSED")


def test_bessel_i0_scaled() raises:
    var result = bessel_i0_scaled(1.0)
    var expected = 0.465759607393445
    assert tolerance(result.val, expected, 1e-10)
    print("test_bessel_i0_scaled: PASSED")


def test_bessel_i1_scaled() raises:
    var result = bessel_i1_scaled(1.0)
    var expected = 0.207910415636708
    assert tolerance(result.val, expected, 1e-10)
    print("test_bessel_i1_scaled: PASSED")


def test_bessel_k0_scaled() raises:
    var result = bessel_k0_scaled(1.0)
    var expected = 0.421598436612131
    assert tolerance(result.val, expected, 1e-10)
    print("test_bessel_k0_scaled: PASSED")


def test_bessel_k1_scaled() raises:
    var result = bessel_k1_scaled(1.0)
    var expected = 0.601948085157574
    assert tolerance(result.val, expected, 1e-10)
    print("test_bessel_k1_scaled: PASSED")


def test_gamma_1() raises:
    var result = gamma(1.0)
    var expected = 1.0
    assert tolerance(result.val, expected, 1e-10)
    print("test_gamma_1: PASSED")


def test_gamma_2() raises:
    var result = gamma(2.0)
    var expected = 1.0
    assert tolerance(result.val, expected, 1e-10)
    print("test_gamma_2: PASSED")


def test_gamma_5() raises:
    var result = gamma(5.0)
    var expected = 24.0
    assert tolerance(result.val, expected, 1e-10)
    print("test_gamma_5: PASSED")


def test_lngamma_1() raises:
    var result = lngamma(1.0)
    var expected = 0.0
    assert tolerance(result.val, expected, 1e-10)
    print("test_lngamma_1: PASSED")


def test_lngamma_2() raises:
    var result = lngamma(2.0)
    var expected = 0.0
    assert tolerance(result.val, expected, 1e-10)
    print("test_lngamma_2: PASSED")


def test_lngamma_half() raises:
    var result = lngamma(0.5)
    var expected = 0.572364942924700
    assert tolerance(result.val, expected, 1e-10)
    print("test_lngamma_half: PASSED")


def test_gammastar_1() raises:
    var result = gammastar(1.0)
    var expected = 1.0
    assert tolerance(result.val, expected, 1e-10)
    print("test_gammastar_1: PASSED")


def test_gammastar_large() raises:
    var result = gammastar(10.0)
    var expected = 1.003892377366556
    assert tolerance(result.val, expected, 1e-10)
    print("test_gammastar_large: PASSED")


def test_factorial_5() raises:
    var result = factorial(5)
    var expected = 120.0
    assert tolerance(result.val, expected, 1e-10)
    print("test_factorial_5: PASSED")


def test_factorial_20() raises:
    var result = factorial(20)
    var expected = 2432902008176640000.0
    assert tolerance(result.val, expected, 1e-10)
    print("test_factorial_20: PASSED")


def test_doublefactorial_5() raises:
    var result = double_factorial(5)
    var expected = 15.0
    assert tolerance(result.val, expected, 1e-10)
    print("test_doublefactorial_5: PASSED")


def test_ln_factorial_5() raises:
    var result = ln_factorial(5)
    var expected = 4.787491742782046
    assert tolerance(result.val, expected, 1e-10)
    print("test_ln_factorial_5: PASSED")


def test_beta_1_1() raises:
    var result = beta(1.0, 1.0)
    var expected = 1.0
    assert tolerance(result.val, expected, 1e-10)
    print("test_beta_1_1: PASSED")


def test_beta_2_3() raises:
    var result = beta(2.0, 3.0)
    var expected = 0.083333333333333
    assert tolerance(result.val, expected, 1e-10)
    print("test_beta_2_3: PASSED")


def test_lnbeta_2_3() raises:
    var result = lnbeta(2.0, 3.0)
    var expected = -2.484906649788000
    assert tolerance(result.val, expected, 1e-10)
    print("test_lnbeta_2_3: PASSED")


def test_erf_zero() raises:
    var result = erf(0.0)
    var expected = 0.0
    assert tolerance(result.val, expected, 1e-10)
    print("test_erf_zero: PASSED")


def test_erf_1() raises:
    var result = erf(1.0)
    var expected = 0.842700792949715
    assert tolerance(result.val, expected, 1e-10)
    print("test_erf_1: PASSED")


def test_erf_neg1() raises:
    var result = erf(-1.0)
    var expected = -0.842700792949715
    assert tolerance(result.val, expected, 1e-10)
    print("test_erf_neg1: PASSED")


def test_erfc_1() raises:
    var result = erfc(1.0)
    var expected = 0.157299207050285
    assert tolerance(result.val, expected, 1e-10)
    print("test_erfc_1: PASSED")


def test_erfc_large() raises:
    var result = erfc(10.0)
    var expected = 1.539338843629752e-46
    assert tolerance(result.val, expected, 1e-10)
    print("test_erfc_large: PASSED")


def test_log_erfc_1() raises:
    var result = log_erfc(1.0)
    var expected = -1.847159059513981
    assert tolerance(result.val, expected, 1e-10)
    print("test_log_erfc_1: PASSED")


def test_erf_Z_0() raises:
    var result = erf_Z(0.0)
    var expected = 0.398942280401433
    assert tolerance(result.val, expected, 1e-10)
    print("test_erf_Z_0: PASSED")


def test_erf_Q_0() raises:
    var result = erf_Q(0.0)
    var expected = 0.5
    assert tolerance(result.val, expected, 1e-10)
    print("test_erf_Q_0: PASSED")


def test_legendre_P1() raises:
    var result = legendre_P1(0.5)
    var expected = 0.5
    assert tolerance(result.val, expected, 1e-10)
    print("test_legendre_P1: PASSED")


def test_legendre_P2() raises:
    var result = legendre_P2(0.5)
    var expected = -0.125
    assert tolerance(result.val, expected, 1e-10)
    print("test_legendre_P2: PASSED")


def test_legendre_P3() raises:
    var result = legendre_P3(0.5)
    var expected = 0.625
    assert tolerance(result.val, expected, 1e-10)
    print("test_legendre_P3: PASSED")


def test_legendre_Pl_0() raises:
    var result = legendre_Pl(0, 0.5)
    var expected = 1.0
    assert tolerance(result.val, expected, 1e-10)
    print("test_legendre_Pl_0: PASSED")


def test_legendre_Pl_1() raises:
    var result = legendre_Pl(1, 0.5)
    var expected = 0.5
    assert tolerance(result.val, expected, 1e-10)
    print("test_legendre_Pl_1: PASSED")


def test_legendre_Pl_2() raises:
    var result = legendre_Pl(2, 0.5)
    var expected = -0.125
    assert tolerance(result.val, expected, 1e-10)
    print("test_legendre_Pl_2: PASSED")


def test_legendre_Pl_10() raises:
    var result = legendre_Pl(10, 0.5)
    var expected = -0.1882286071777344
    assert tolerance(result.val, expected, 1e-10)
    print("test_legendre_Pl_10: PASSED")


def test_legendre_Pl_x1() raises:
    var result = legendre_Pl(10, 1.0)
    var expected = 1.0
    assert tolerance(result.val, expected, 1e-10)
    print("test_legendre_Pl_x1: PASSED")


def test_legendre_Pl_xneg1() raises:
    var result = legendre_Pl(10, -1.0)
    var expected = 1.0
    assert tolerance(result.val, expected, 1e-10)
    print("test_legendre_Pl_xneg1: PASSED")


def test_legendre_Pl_1000() raises:
    var result = legendre_Pl(1000, 0.5)
    var expected = 0.02522501817709829
    assert tolerance(result.val, expected, 1e-4)
    print("test_legendre_Pl_1000: PASSED")


def test_legendre_Pl_100000() raises:
    var result = legendre_Pl(100000, 0.5)
    var expected = 0.02522501817709829
    assert tolerance(result.val, expected, 1e-2)
    print("test_legendre_Pl_100000: PASSED")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("All sf tests PASSED")
