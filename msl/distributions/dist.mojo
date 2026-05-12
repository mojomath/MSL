# SPDX-License-Identifier: GPL-3.0-or-later

# ===----------------------------------------------------------------------=== #
# MSL (Mojo Scientific Library)
#
# Derived from GNU Scientific Library (GSL)
# Original file: gsl_randist.h
#
# Original authors:
# Copyright (C) 1996–2007 James Theiler, Brian Gough
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
Random distributions.
Pure Mojo implementation of common probability distributions.
"""

from std.math import exp, log, log1p, sqrt, cos, sin, abs, tan

from msl.core.const import MSL_PI, MSL_FLT_EPSILON
from msl.rng import RNG, RNGAlgorithm, MT19937
from msl.sf.gamma import lngamma_lanczos

# ===----------------------------------------------------------------------=== #
# Gaussian
# ===----------------------------------------------------------------------=== #


def gaussian[T: RNGAlgorithm](mut rng: RNG[T], sigma: Float64) -> Float64:
    var x1 = rng.uniform()
    var x2 = rng.uniform()

    while x1 <= MSL_FLT_EPSILON:
        x1 = rng.uniform()
        x2 = rng.uniform()

    var r = sqrt(-2.0 * log(x1)) * sigma
    return r * cos(2.0 * MSL_PI * x2)


def gaussian_pdf(x: Float64, sigma: Float64) -> Float64:
    return exp(-0.5 * (x / sigma) * (x / sigma)) / (sigma * sqrt(2.0 * MSL_PI))


# ===----------------------------------------------------------------------=== #
# Uniform
# ===----------------------------------------------------------------------=== #


def uniform[
    T: RNGAlgorithm
](mut rng: RNG[T], a: Float64, b: Float64) -> Float64:
    return a + (b - a) * rng.uniform()


def uniform_pdf(x: Float64, a: Float64, b: Float64) -> Float64:
    if x < a or x > b:
        return 0.0
    return 1.0 / (b - a)


# ===----------------------------------------------------------------------=== #
# Exponential
# ===----------------------------------------------------------------------=== #


def exponential[T: RNGAlgorithm](mut rng: RNG[T], mu: Float64) -> Float64:
    return -mu * log(rng.uniform())


def exponential_pdf(x: Float64, mu: Float64) -> Float64:
    if x < 0.0:
        return 0.0
    return exp(-x / mu) / mu


# ===----------------------------------------------------------------------=== #
# Gamma
# ===----------------------------------------------------------------------=== #


def gamma[T: RNGAlgorithm](mut rng: RNG[T], a: Float64, b: Float64) -> Float64:
    if a < 1.0:
        return gamma(rng, a + 1.0, b) * rng.uniform() ** (1.0 / a)

    var d = a - 1.0 / 3.0
    var c = sqrt(1.0 / 9.0 / d)

    while True:
        var x = gaussian(rng, 1.0)
        var v = 1.0 + c * x

        if v <= 0.0:
            continue

        v = v * v * v
        var u = rng.uniform()

        if u < 1.0 - 0.0331 * (x * x) * (x * x):
            return d * v * b

        if log(u) < 0.5 * x * x + d * (1.0 - v + log(v)):
            return d * v * b


def gamma_pdf(x: Float64, a: Float64, b: Float64) -> Float64:
    if x <= 0.0:
        return 0.0
    var z = x / b
    return z ** (a - 1.0) * exp(-z) / (b * _gamma_func(a))


def _gamma_func(x: Float64) -> Float64:
    var xn = x
    var g = 7.0
    var c = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ]

    if xn < 0.5:
        return (
            MSL_PI / (sin(xn + 0.5) * _gamma_func(1.0 - xn)) - log(x) - MSL_PI
        )

    xn -= 1.0
    var a = c[0]
    var t = xn + g + 0.5
    for i in range(1, 9):
        a += c[i] / (xn + Float64(i))
    return log(t) + 0.5 * exp(-t) - 0.918938533204673 + log(a / t)


# ===----------------------------------------------------------------------=== #
# Beta
# ===----------------------------------------------------------------------=== #


def beta[T: RNGAlgorithm](mut rng: RNG[T], a: Float64, b: Float64) -> Float64:
    var x = gamma(rng, a, 1.0)
    var y = gamma(rng, b, 1.0)
    return x / (x + y)


# ===----------------------------------------------------------------------=== #
# Chi-squared
# ===----------------------------------------------------------------------=== #


def chisq[T: RNGAlgorithm](mut rng: RNG[T], nu: Float64) -> Float64:
    return 2.0 * gamma(rng, nu / 2.0, 1.0)


# ===----------------------------------------------------------------------=== #
# Poisson
# ===----------------------------------------------------------------------=== #


def poisson[T: RNGAlgorithm](mut rng: RNG[T], mu: Float64) -> Int:
    """Sample from Poisson(mu) using Knuth's algorithm."""
    var L = exp(-mu)
    var k: Int = 0
    var p: Float64 = 1.0
    while True:
        k += 1
        p *= rng.uniform()
        if p <= L:
            break
    return k - 1


def poisson_pdf(k: Int, mu: Float64) -> Float64:
    if k < 0:
        return 0.0
    var log_k_factorial = log(_gamma_func(Float64(k) + 1.0))
    return exp(Float64(k) * log(mu) - mu - log_k_factorial)


# ===----------------------------------------------------------------------=== #
# Student's t
# ===----------------------------------------------------------------------=== #


def tdist[T: RNGAlgorithm](mut rng: RNG[T], nu: Float64) -> Float64:
    """Sample from Student's t distribution with nu degrees of freedom."""
    if nu <= 2.0:
        var y1 = gaussian(rng, 1.0)
        var y2 = chisq(rng, nu)
        return y1 / sqrt(y2 / nu)
    else:
        while True:
            var y1 = gaussian(rng, 1.0)
            var y2 = exponential(rng, 2.0 / (nu - 2.0))
            var z = y1 * y1 / (nu - 2.0)
            if (1.0 - z) > 0.0 and exp(-y2 - z) <= (1.0 - z):
                return y1 / sqrt((1.0 - 2.0 / nu) * (1.0 - z))


def tdist_pdf(x: Float64, nu: Float64) -> Float64:
    """PDF of Student's t distribution."""
    var lg1 = lngamma_lanczos((nu + 1.0) / 2.0)
    var lg2 = lngamma_lanczos(nu / 2.0)
    return (
        exp(lg1 - lg2)
        / sqrt(MSL_PI * nu)
        * (1.0 + x * x / nu) ** (-(nu + 1.0) / 2.0)
    )


# ===----------------------------------------------------------------------=== #
# Log-normal
# ===----------------------------------------------------------------------=== #


def lognormal[
    T: RNGAlgorithm
](mut rng: RNG[T], zeta: Float64, sigma: Float64) -> Float64:
    """Sample from log-normal distribution with log-mean zeta and log-std sigma.
    """
    var u: Float64
    var v: Float64
    var r2: Float64
    while True:
        u = 2.0 * rng.uniform() - 1.0
        v = 2.0 * rng.uniform() - 1.0
        r2 = u * u + v * v
        if r2 <= 1.0 and r2 != 0.0:
            break
    var normal = u * sqrt(-2.0 * log(r2) / r2)
    return exp(sigma * normal + zeta)


def lognormal_pdf(x: Float64, zeta: Float64, sigma: Float64) -> Float64:
    """PDF of log-normal distribution."""
    if x <= 0.0:
        return 0.0
    var u = (log(x) - zeta) / sigma
    return exp(-0.5 * u * u) / (x * abs(sigma) * sqrt(2.0 * MSL_PI))


# ===----------------------------------------------------------------------=== #
# Weibull
# ===----------------------------------------------------------------------=== #


def weibull[
    T: RNGAlgorithm
](mut rng: RNG[T], a: Float64, b: Float64) -> Float64:
    """Sample from Weibull distribution with scale a and shape b."""
    return a * (-log(rng.uniform_pos())) ** (1.0 / b)


def weibull_pdf(x: Float64, a: Float64, b: Float64) -> Float64:
    """PDF of Weibull distribution."""
    if x < 0.0:
        return 0.0
    if x == 0.0:
        return 1.0 / a if b == 1.0 else 0.0
    return (b / a) * exp(-((x / a) ** b) + (b - 1.0) * log(x / a))


# ===----------------------------------------------------------------------=== #
# Binomial
# ===----------------------------------------------------------------------=== #


def binomial[T: RNGAlgorithm](mut rng: RNG[T], p: Float64, n: Int) -> Int:
    """Sample from Binomial(n, p) via Knuth beta-splitting."""
    if p == 0.0:
        return 0
    if p == 1.0:
        return n
    var k: Int = 0
    var remaining = n
    var p_rem = p
    while remaining > 10:
        var a = 1 + remaining // 2
        var b_param = 1 + remaining - a
        var x = beta(rng, Float64(a), Float64(b_param))
        if x >= p_rem:
            remaining = a - 1
        else:
            k += a
            remaining = b_param - 1
            p_rem = (p_rem - x) / (1.0 - x)
    for _ in range(remaining):
        if rng.uniform() < p_rem:
            k += 1
    return k


def binomial_pdf(k: Int, p: Float64, n: Int) -> Float64:
    """PMF of Binomial(n, p)."""
    if k < 0 or k > n:
        return 0.0
    if p == 0.0:
        return 1.0 if k == 0 else 0.0
    if p == 1.0:
        return 1.0 if k == n else 0.0
    var fk = Float64(k)
    var fln = Float64(n)
    var lnchoose = (
        lngamma_lanczos(fln + 1.0)
        - lngamma_lanczos(fk + 1.0)
        - lngamma_lanczos(fln - fk + 1.0)
    )
    return exp(lnchoose + fk * log(p) + (fln - fk) * log1p(-p))


# ===----------------------------------------------------------------------=== #
# Negative Binomial
# ===----------------------------------------------------------------------=== #


def negative_binomial[
    T: RNGAlgorithm
](mut rng: RNG[T], p: Float64, n: Float64) -> Int:
    """Sample from Negative Binomial(n, p) via Gamma-Poisson mixture."""
    var x = gamma(rng, n, 1.0)
    return poisson(rng, x * (1.0 - p) / p)


def negative_binomial_pdf(k: Int, p: Float64, n: Float64) -> Float64:
    """PMF of Negative Binomial(n, p)."""
    if k < 0:
        return 0.0
    var fk = Float64(k)
    var log_p = log(p)
    return exp(
        lngamma_lanczos(n + fk)
        - lngamma_lanczos(n)
        - lngamma_lanczos(fk + 1.0)
        + n * log_p
        + fk * log1p(-p)
    )


# ===----------------------------------------------------------------------=== #
# Cauchy
# ===----------------------------------------------------------------------=== #


def cauchy[T: RNGAlgorithm](mut rng: RNG[T], a: Float64) -> Float64:
    """Sample from Cauchy distribution with scale a."""
    var u: Float64
    while True:
        u = rng.uniform()
        if u != 0.5:
            break
    return a * tan(MSL_PI * u)


def cauchy_pdf(x: Float64, a: Float64) -> Float64:
    """PDF of Cauchy distribution."""
    var u = x / a
    return 1.0 / (MSL_PI * a * (1.0 + u * u))


# ===----------------------------------------------------------------------=== #
# Laplace (double exponential)
# ===----------------------------------------------------------------------=== #


def laplace[T: RNGAlgorithm](mut rng: RNG[T], a: Float64) -> Float64:
    """Sample from Laplace distribution with scale a."""
    var u: Float64
    while True:
        u = 2.0 * rng.uniform() - 1.0
        if u != 0.0:
            break
    if u < 0.0:
        return a * log(-u)
    else:
        return -a * log(u)


def laplace_pdf(x: Float64, a: Float64) -> Float64:
    """PDF of Laplace distribution."""
    return exp(-abs(x) / a) / (2.0 * a)
