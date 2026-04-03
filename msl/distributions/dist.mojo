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

from std.math import exp, log, sqrt, cos, sin

from msl.core.const import MSL_PI, MSL_FLT_EPSILON
from msl.rng import RNG

# ===----------------------------------------------------------------------=== #
# Gaussian
# ===----------------------------------------------------------------------=== #

fn gaussian(mut rng: RNG, sigma: Float64) -> Float64:
    var x1 = rng.uniform()
    var x2 = rng.uniform()

    while x1 <= MSL_FLT_EPSILON:
        x1 = rng.uniform()
        x2 = rng.uniform()

    var r = sqrt(-2.0 * log(x1)) * sigma
    return r * cos(2.0 * MSL_PI * x2)

fn gaussian_pdf(x: Float64, sigma: Float64) -> Float64:
    return exp(-0.5 * (x / sigma) * (x / sigma)) / (sigma * sqrt(2.0 * MSL_PI))

# ===----------------------------------------------------------------------=== #
# Uniform
# ===----------------------------------------------------------------------=== #

fn uniform(mut rng: RNG, a: Float64, b: Float64) -> Float64:
    return a + (b - a) * rng.uniform()

fn uniform_pdf(x: Float64, a: Float64, b: Float64) -> Float64:
    if x < a or x > b:
        return 0.0
    return 1.0 / (b - a)

# ===----------------------------------------------------------------------=== #
# Exponential
# ===----------------------------------------------------------------------=== #

fn exponential(mut rng: RNG, mu: Float64) -> Float64:
    return -mu * log(rng.uniform())

fn exponential_pdf(x: Float64, mu: Float64) -> Float64:
    if x < 0.0:
        return 0.0
    return exp(-x / mu) / mu

# ===----------------------------------------------------------------------=== #
# Gamma
# ===----------------------------------------------------------------------=== #

fn gamma(mut rng: RNG, a: Float64, b: Float64) -> Float64:
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

fn gamma_pdf(x: Float64, a: Float64, b: Float64) -> Float64:
    if x <= 0.0:
        return 0.0
    var z = x / b
    return z ** (a - 1.0) * exp(-z) / (b * _gamma_func(a))

fn _gamma_func(x: Float64) -> Float64:
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
        return MSL_PI / (sin(xn + 0.5) * _gamma_func(1.0 - xn)) - log(x) - MSL_PI

    xn -= 1.0
    var a = c[0]
    var t = xn + g + 0.5
    for i in range(1, 10):
        a += c[i] / (xn + Float64(i))
    return (log(t) + 0.5 * exp(-t) - 0.918938533204673 + log(a / t))

# ===----------------------------------------------------------------------=== #
# Beta
# ===----------------------------------------------------------------------=== #

fn beta(mut rng: RNG, a: Float64, b: Float64) -> Float64:
    var x = gamma(rng, a, 1.0)
    var y = gamma(rng, b, 1.0)
    return x / (x + y)

# ===----------------------------------------------------------------------=== #
# Chi-squared
# ===----------------------------------------------------------------------=== #

fn chisq(mut rng: RNG, nu: Float64) -> Float64:
    return 2.0 * gamma(rng, nu / 2.0, 1.0)

# ===----------------------------------------------------------------------=== #
# Poisson
# ===----------------------------------------------------------------------=== #

fn poisson(mut rng: RNG, mu: Float64) -> Int:
    if mu < 30.0:
        var em: Float64 = -1.0
        var sum: Float64 = 0.0
        while True:
            em += 1.0
            sum += exp(-(em + 1.0) / mu) * rng.uniform()
            if sum > 1.0:
                break
        return Int(em)
    else:
        var L = exp(-mu)
        var k: Int = 0
        while True:
            var p = L
            var u = rng.uniform()
            while u > p:
                k += 1
                p *= mu / Float64(k)
                u = rng.uniform()
            if u <= p:
                break
        return k

fn poisson_pdf(k: Int, mu: Float64) -> Float64:
    if k < 0:
        return 0.0
    # Use log for numerical stability: P = exp(k*log(mu) - mu - log(k!))
    # k! = Gamma(k+1), so log(k!) = log(gamma_func(k+1))
    var log_k_factorial = log(_gamma_func(Float64(k) + 1.0))
    return exp(Float64(k) * log(mu) - mu - log_k_factorial)
