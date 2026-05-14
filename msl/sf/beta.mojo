# SPDX-License-Identifier: GPL-3.0-or-later

# ===----------------------------------------------------------------------=== #
# MSL (Mojo Scientific Library)
#
# Derived from GNU Scientific Library (GSL)
# Original file: specfunc/bessel_J0.c, bessel_J1.c, etc.
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
Beta function and log beta function.
"""

from std.math import exp

from msl.core.const import MSL_DBL_EPSILON, MSL_SQRT_DBL_EPSILON, MSL_PI
from msl.sf.result import SFSResult
from msl.core.errno import MSL_EDOM

# ===----------------------------------------------------------------------=== #
# Beta function
# ===----------------------------------------------------------------------=== #


def _beta(a: Float64, b: Float64) -> SFSResult:
    var result = SFSResult()

    if a <= 0.0 or b <= 0.0:
        result.errno = MSL_EDOM
        return result^

    if a > b:
        return _beta(b, a)

    if a + b < 170.0:
        result.val = exp(lngamma(a + b).val - lngamma(a).val - lngamma(b).val)
        result.err = result.val * (
            lngamma(a + b).err + lngamma(a).err + lngamma(b).err
        )
    else:
        var lg_a = lngamma(a)
        var lg_b = lngamma(b)
        var lg_ab = lngamma(a + b)

        if lg_ab.val - lg_a.val - lg_b.val < -700.0:
            result.val = 0.0
            result.err = MSL_DBL_EPSILON
        else:
            result.val = exp(lg_ab.val - lg_a.val - lg_b.val)
            result.err = result.val * (lg_ab.err + lg_a.err + lg_b.err)

    result.err += MSL_DBL_EPSILON * abs(result.val)
    return result^


# ===----------------------------------------------------------------------=== #
# Log Beta function
# ===----------------------------------------------------------------------=== #


def _lnbeta(a: Float64, b: Float64) -> SFSResult:
    var result = SFSResult()

    if a <= 0.0 or b <= 0.0:
        result.errno = MSL_EDOM
        return result^

    var lg_a = lngamma(a)
    var lg_b = lngamma(b)
    var lg_ab = lngamma(a + b)

    result.val = lg_ab.val - lg_a.val - lg_b.val
    result.err = lg_ab.err + lg_a.err + lg_b.err
    result.err += MSL_DBL_EPSILON * abs(result.val)
    return result^


# ===----------------------------------------------------------------------=== #
# Public API
# ===----------------------------------------------------------------------=== #


def beta(a: Float64, b: Float64) -> SFSResult:
    return _beta(a, b)


def lnbeta(a: Float64, b: Float64) -> SFSResult:
    return _lnbeta(a, b)
