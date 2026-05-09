# SPDX-License-Identifier: GPL-3.0-or-later

# ===----------------------------------------------------------------------=== #
# MSL (Mojo Scientific Library)
#
# Derived from GNU Scientific Library (GSL)
# Original file: specfunc/gamma_inc.c
#
# Original authors:
# Copyright (C) 1996-2007 Gerard Jungman
#
# Modifications:
# Copyright (C) 2026 Shivasankar K.A.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
# ===----------------------------------------------------------------------=== #
"""
Regularized incomplete gamma functions P(a,x) and Q(a,x).

  P(a,x) = 1/Gamma(a) * integral_0^x  t^(a-1) * exp(-t) dt
  Q(a,x) = 1 - P(a,x)
"""

from std.math import exp, log, sqrt, abs, erfc, log1p

from msl.core.const import MSL_DBL_EPSILON, MSL_PI
from msl.core.errno import GSL_EDOM
from msl.sf.result import SFSResult
from msl.sf.gamma import lngamma_lanczos, _gammastar


# ===----------------------------------------------------------------------=== #
# Helpers
# ===----------------------------------------------------------------------=== #


def _log1pmx(x: Float64) -> Float64:
    """log(1+x) - x, accurate for small x."""
    if abs(x) < 0.5:
        var t = x / (2.0 + x)
        var t2 = t * t
        return (
            -2.0
            * t
            * t2
            * (1.0 / 3.0 + t2 * (1.0 / 5.0 + t2 * (1.0 / 7.0 + t2 / 9.0)))
        )
    else:
        return log1p(x) - x


def _D_factor(a: Float64, x: Float64) -> Float64:
    """Compute D(a,x) = x^a * exp(-x) / Gamma(a+1), the common prefactor."""
    if a < 10.0:
        return exp(a * log(x) - x - lngamma_lanczos(a + 1.0))
    else:
        var u = x / a
        var ln_term = log(u) - u + 1.0
        return exp(a * ln_term) / (sqrt(2.0 * MSL_PI * a) * _gammastar(a).val)


# ===----------------------------------------------------------------------=== #
# P-series: P(a,x) = D(a,x) * sum_{n=0}^inf x^n / (a+1)(a+2)...(a+n)
# ===----------------------------------------------------------------------=== #


def _gamma_inc_P_series(a: Float64, x: Float64) -> Float64:
    var D = _D_factor(a, x)
    var sum: Float64 = 1.0
    var term: Float64 = 1.0
    for n in range(1, 500):
        term *= x / (a + Float64(n))
        sum += term
        if abs(term) < abs(sum) * MSL_DBL_EPSILON:
            break
    return D * sum


# ===----------------------------------------------------------------------=== #
# Q continued fraction via modified Lentz
# ===----------------------------------------------------------------------=== #


def _gamma_inc_Q_CF(a: Float64, x: Float64) -> Float64:
    var D = _D_factor(a, x)
    var FPMIN: Float64 = 1.0e-300

    # Seed the Lentz recurrence
    var b = x + 1.0 - a
    var c = 1.0 / FPMIN
    var d = 1.0 / b
    var h = d

    for i in range(1, 500):
        var fi = Float64(i)
        var an = -fi * (fi - a)
        b += 2.0
        d = an * d + b
        if abs(d) < FPMIN:
            d = FPMIN
        c = b + an / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        var delta = d * c
        h *= delta
        if abs(delta - 1.0) < MSL_DBL_EPSILON:
            break

    return exp(-x + a * log(x) - lngamma_lanczos(a)) * h


# ===----------------------------------------------------------------------=== #
# Q large-x asymptotic
# ===----------------------------------------------------------------------=== #


def _gamma_inc_Q_large_x(a: Float64, x: Float64) -> Float64:
    var D = _D_factor(a, x)
    var sum: Float64 = 1.0
    var term: Float64 = 1.0
    for n in range(1, 500):
        var prev = term
        term *= (a - Float64(n)) / x
        if abs(term) > abs(prev):
            break
        if abs(term) < abs(sum) * MSL_DBL_EPSILON:
            break
        sum += term
    return D * (a / x) * sum


# ===----------------------------------------------------------------------=== #
# Q uniform asymptotic (Temme) for large a
# ===----------------------------------------------------------------------=== #


def _gamma_inc_Q_asymp_unif(a: Float64, x: Float64) -> Float64:
    var eps = (x - a) / a
    var ln_term = _log1pmx(eps)
    var eta: Float64
    if ln_term < 0.0:
        eta = sqrt(-2.0 * ln_term)
    else:
        eta = -sqrt(-2.0 * ln_term)
    if eps < 0.0:
        eta = -abs(eta)

    var q_erfc = 0.5 * erfc(eta * sqrt(a * 0.5))

    var c0: Float64
    var c1: Float64
    if abs(eps) < 1.0e-3:
        c0 = -1.0 / 3.0 + eps * (
            1.0 / 12.0
            - eps
            * (23.0 / 540.0 - eps * (353.0 / 12960.0 - eps * 589.0 / 30240.0))
        )
        c1 = -1.0 / 540.0 - eps / 288.0
    else:
        var rt_term = sqrt(-2.0 * ln_term / (eps * eps))
        var lam = x / a
        c0 = (1.0 - 1.0 / rt_term) / eps
        c1 = -(
            eta * eta * eta * (lam * lam + 10.0 * lam + 1.0)
            - 12.0 * eps * eps * eps
        ) / (12.0 * eta * eta * eta * eps * eps * eps)

    var R = exp(-0.5 * a * eta * eta) / sqrt(2.0 * MSL_PI * a) * (c0 + c1 / a)
    return q_erfc + R


# ===----------------------------------------------------------------------=== #
# Q small-a series
# ===----------------------------------------------------------------------=== #

comptime _EULER: Float64 = 0.5772156649015328606065120900824024


def _gamma_inc_Q_series(a: Float64, x: Float64) -> Float64:
    var lnx = log(x)
    var el = _EULER + lnx
    var c1 = -el
    var c2 = MSL_PI * MSL_PI / 12.0 - el * el * 0.5
    var c3 = (
        el * (MSL_PI * MSL_PI / 12.0 - el * el / 6.0)
        + (-2.404113806319188570799476) / 6.0
    )
    var c4 = -0.00138888888888888889 + el * (
        -0.0416666666666667
        + el * (-0.0833333333333333 + el * (-0.0416666666666667 + el * 0.0))
    )
    var c5 = -0.000952380952380952 + el * (
        -0.00277777777777778
        + el
        * (
            0.00833333333333333
            + el * (0.0166666666666667 + el * (0.00416666666666667))
        )
    )

    var term1 = a * (c1 + a * (c2 + a * (c3 + a * (c4 + a * c5))))

    # term2: (1 - term1) * a/(a+1) * x * S
    var sum: Float64 = 1.0
    var t: Float64 = 1.0
    for n in range(1, 200):
        var flt_n = Float64(n)
        t *= -x / (flt_n + 1.0)
        sum += (a + 1.0) / (a + flt_n + 1.0) * t
        if abs(t) < abs(sum) * MSL_DBL_EPSILON:
            break
    var term2 = (1.0 - term1) * a / (a + 1.0) * x * sum
    return term1 + term2


# ===----------------------------------------------------------------------=== #
# Public API
# ===----------------------------------------------------------------------=== #


def gamma_inc_Q(a: Float64, x: Float64) -> SFSResult:
    """Regularized upper incomplete gamma Q(a,x) = 1 - P(a,x).

    Args:
        a: Shape parameter, a > 0.
        x: Integration limit, x >= 0.

    Returns:
        SFSResult with Q(a,x) in [0,1].
    """
    var result = SFSResult()
    if a <= 0.0 or x < 0.0:
        result.errno = GSL_EDOM
        return result^

    if x == 0.0:
        result.val = 1.0
        return result^

    if a == 0.0:
        result.val = 0.0
        return result^

    var val: Float64
    if x <= 0.5 * a:
        val = 1.0 - _gamma_inc_P_series(a, x)
    elif a >= 1.0e6 and (x - a) * (x - a) < a:
        val = _gamma_inc_Q_asymp_unif(a, x)
    elif a < 0.2 and x < 5.0:
        val = _gamma_inc_Q_series(a, x)
    elif a <= x and x <= 1.0e6:
        val = _gamma_inc_Q_CF(a, x)
    elif a <= x and x > 1.0e6:
        val = _gamma_inc_Q_large_x(a, x)
    elif x > a - sqrt(a):
        val = _gamma_inc_Q_CF(a, x)
    else:
        val = 1.0 - _gamma_inc_P_series(a, x)

    result.val = val
    result.err = MSL_DBL_EPSILON * abs(result.val)
    return result^


def gamma_inc_P(a: Float64, x: Float64) -> SFSResult:
    """Regularized lower incomplete gamma P(a,x).

    Args:
        a: Shape parameter, a > 0.
        x: Integration limit, x >= 0.

    Returns:
        SFSResult with P(a,x) in [0,1].
    """
    var result = SFSResult()
    if a <= 0.0 or x < 0.0:
        result.errno = GSL_EDOM
        return result^

    if x == 0.0:
        result.val = 0.0
        return result^

    var val: Float64
    if x < 20.0 or x < 0.5 * a:
        val = _gamma_inc_P_series(a, x)
    elif a > 1.0e6 and (x - a) * (x - a) < a:
        val = 1.0 - _gamma_inc_Q_asymp_unif(a, x)
    elif a <= x and a > 0.2 * x:
        val = 1.0 - _gamma_inc_Q_CF(a, x)
    elif a <= x and a <= 0.2 * x:
        val = 1.0 - _gamma_inc_Q_large_x(a, x)
    elif (x - a) * (x - a) < a:
        val = 1.0 - _gamma_inc_Q_CF(a, x)
    else:
        val = _gamma_inc_P_series(a, x)

    result.val = val
    result.err = MSL_DBL_EPSILON * abs(result.val)
    return result^


def gamma_inc(a: Float64, x: Float64) -> SFSResult:
    """Non-regularized incomplete gamma Gamma(a,x) = Gamma(a) * Q(a,x).

    Args:
        a: Shape parameter, a > 0.
        x: Integration limit, x >= 0.

    Returns:
        SFSResult with Gamma(a,x).
    """
    from msl.sf.gamma import lngamma

    var result = gamma_inc_Q(a, x)
    if result.errno != 0:
        return result^
    var lg = lngamma(a)
    result.val = result.val * exp(lg.val)
    result.err = result.err * exp(lg.val) + abs(result.val) * lg.err
    return result^
