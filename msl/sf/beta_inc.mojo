# SPDX-License-Identifier: GPL-3.0-or-later

# ===----------------------------------------------------------------------=== #
# MSL (Mojo Scientific Library)
#
# Derived from GNU Scientific Library (GSL)
# Original file: specfunc/beta_inc.c
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
Regularized incomplete beta function I_x(a,b).

  beta_inc(a, b, x) = I_x(a,b)
                    = B(x; a, b) / B(a, b)
                    = integral_0^x t^(a-1) (1-t)^(b-1) dt / B(a,b)
"""

from std.math import exp, log, log1p, abs

from msl.core.const import MSL_DBL_EPSILON
from msl.core.errno import GSL_EDOM
from msl.sf.result import SFSResult
from msl.sf.gamma import lngamma_lanczos


# ===----------------------------------------------------------------------=== #
# Continued fraction (Lentz) for I_x(a,b)
# ===----------------------------------------------------------------------=== #


def _beta_inc_CF(a: Float64, b: Float64, x: Float64) -> Float64:
    """Evaluate I_x(a,b) via continued fraction (convergent when x < (a+1)/(a+b+2)).
    """
    var FPMIN: Float64 = 1.0e-300

    # Prefactor: x^a * (1-x)^b / B(a,b)
    var ln_pre = (
        lngamma_lanczos(a + b)
        - lngamma_lanczos(a)
        - lngamma_lanczos(b)
        + a * log(x)
        + b * log1p(-x)
    )
    var pre = exp(ln_pre)

    # Lentz modified continued fraction
    var b0 = x + 1.0 - a - (a + b) * x / (a + 1.0)
    var c = 1.0 / FPMIN
    var d = 1.0 / b0 if abs(b0) > FPMIN else FPMIN
    var h = d

    for m in range(1, 500):
        var fm = Float64(m)
        # Even step
        var d_even = fm * (b - fm) * x / ((a + 2.0 * fm - 1.0) * (a + 2.0 * fm))
        b0 += 2.0
        d = d_even * d + b0
        if abs(d) < FPMIN:
            d = FPMIN
        c = b0 + d_even / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c

        # Odd step
        var d_odd = (
            -(a + fm)
            * (a + b + fm)
            * x
            / ((a + 2.0 * fm) * (a + 2.0 * fm + 1.0))
        )
        b0 += 2.0
        d = d_odd * d + b0
        if abs(d) < FPMIN:
            d = FPMIN
        c = b0 + d_odd / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        var delta = d * c
        h *= delta
        if abs(delta - 1.0) < MSL_DBL_EPSILON:
            break

    return pre * h / a


# ===----------------------------------------------------------------------=== #
# Public API
# ===----------------------------------------------------------------------=== #


def beta_inc(a: Float64, b: Float64, x: Float64) -> SFSResult:
    """Regularized incomplete beta function I_x(a,b).

    Args:
        a: First shape parameter, a > 0.
        b: Second shape parameter, b > 0.
        x: Upper limit, 0 <= x <= 1.

    Returns:
        SFSResult with I_x(a,b) in [0,1].
    """
    var result = SFSResult()
    if a <= 0.0 or b <= 0.0 or x < 0.0 or x > 1.0:
        result.errno = GSL_EDOM
        return result^

    if x == 0.0:
        result.val = 0.0
        return result^

    if x == 1.0:
        result.val = 1.0
        return result^

    var threshold = (a + 1.0) / (a + b + 2.0)
    var val: Float64
    if x < threshold:
        val = _beta_inc_CF(a, b, x)
    else:
        # Symmetry: I_x(a,b) = 1 - I_{1-x}(b,a)
        val = 1.0 - _beta_inc_CF(b, a, 1.0 - x)

    result.val = val
    result.err = MSL_DBL_EPSILON * abs(result.val)
    return result^
