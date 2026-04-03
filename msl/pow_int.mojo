# SPDX-License-Identifier: GPL-3.0-or-later

# ===----------------------------------------------------------------------=== #
# MSL (Mojo Scientific Library)
#
# Derived from GNU Scientific Library (GSL)
# Original file: gsl_pow_int.h
#
# Original authors:
# Copyright (C) 1996–2007 Gerard Jungman, Brian Gough
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
Integer power functions.

Efficient computation of integer powers.
"""


# ===----------------------------------------------------------------------=== #
# Integer powers
# ===----------------------------------------------------------------------=== #


def msl_pow_2(x: Float64) -> Float64:
    """Return x^2."""
    return x * x


def msl_pow_3(x: Float64) -> Float64:
    """Return x^3."""
    return x * x * x


def msl_pow_4(x: Float64) -> Float64:
    """Return x^4."""
    return x * x * x * x


def msl_pow_5(x: Float64) -> Float64:
    """Return x^5."""
    return x * x * x * x * x


def msl_pow_6(x: Float64) -> Float64:
    """Return x^6."""
    return x * x * x * x * x * x


def msl_pow_7(x: Float64) -> Float64:
    """Return x^7."""
    return x * x * x * x * x * x * x


def msl_pow_8(x: Float64) -> Float64:
    """Return x^8."""
    return x * x * x * x * x * x * x * x


def msl_pow_9(x: Float64) -> Float64:
    """Return x^9."""
    return x * x * x * x * x * x * x * x * x


def msl_pow_10(x: Float64) -> Float64:
    """Return x^10."""
    return x * x * x * x * x * x * x * x * x * x


def msl_pow_int(x: Float64, n: Int) -> Float64:
    """Return x^n for any integer n.

    Uses binary exponentiation for efficiency.
    """
    var xn = x
    var nn = n
    if nn == 0:
        return 1.0
    if nn < 0:
        xn = 1.0 / xn
        nn = -nn

    var result: Float64 = 1.0
    var base = xn
    while nn > 0:
        if nn % 2 == 1:
            result *= base
        base *= base
        nn //= 2
    return result
