# SPDX-License-Identifier: GPL-3.0-or-later

# ===----------------------------------------------------------------------=== #
# MSL (Mojo Scientific Library)
#
# Derived from GNU Scientific Library (GSL)
# Original file: specfunc/psi.c
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
Digamma (psi) function.

psi(x) = d/dx ln(Gamma(x))
"""

from std.math import log, sqrt, abs, floor
from std.collections import InlineArray

from msl.core.const import MSL_DBL_EPSILON, MSL_PI
from msl.core.errno import MSL_EDOM
from msl.sf.result import SFSResult
from msl.sf.gamma import cheb_eval

# Euler-Mascheroni constant
comptime MSL_EULER: Float64 = 0.5772156649015328606065120900824024

# ===----------------------------------------------------------------------=== #
# Chebyshev data: psi on [0,1], t = 2x - 1
# ===----------------------------------------------------------------------=== #

comptime _psics_data: InlineArray[Float64, 23] = [
    -0.038057080835217922,
    0.491415393029387130,
    -0.056815747821244730,
    0.008357821225914313,
    -0.001333232857994342,
    0.000220313287069308,
    -0.000037040238178456,
    0.000006283793654854,
    -0.000001071263908506,
    0.000000183128394654,
    -0.000000031353509361,
    0.000000005372808776,
    -0.000000000921168141,
    0.000000000157981265,
    -0.000000000027098646,
    0.000000000004648722,
    -0.000000000000797527,
    0.000000000000136827,
    -0.000000000000023475,
    0.000000000000004027,
    -0.000000000000000691,
    0.000000000000000118,
    -0.000000000000000020,
]

# ===----------------------------------------------------------------------=== #
# Chebyshev data: asymptotic psi on [2,inf), t = 8/x^2 - 1
# ===----------------------------------------------------------------------=== #

comptime _apsics_data: InlineArray[Float64, 16] = [
    -0.0204749044678185,
    -0.0101801271534859,
    0.0000559718725387,
    -0.0000012917176570,
    0.0000000572858606,
    -0.0000000038213539,
    0.0000000003397434,
    -0.0000000000374838,
    0.0000000000048990,
    -0.0000000000007344,
    0.0000000000001233,
    -0.0000000000000228,
    0.0000000000000045,
    -0.0000000000000009,
    0.0000000000000002,
    -0.0000000000000000,
]


# ===----------------------------------------------------------------------=== #
# Internal implementation
# ===----------------------------------------------------------------------=== #


def _psi_x(x: Float64) -> Float64:
    """Digamma for x > 0."""
    if x >= 2.0:
        var t = 8.0 / (x * x) - 1.0
        return (
            log(x) - 0.5 / x + cheb_eval[16](_apsics_data, 15, -1.0, 1.0, t).val
        )
    elif x >= 1.0:
        return cheb_eval[23](
            _psics_data, 22, -1.0, 1.0, 2.0 * (x - 1.0) - 1.0
        ).val
    else:
        return (
            -1.0 / x
            + cheb_eval[23](_psics_data, 22, -1.0, 1.0, 2.0 * x - 1.0).val
        )


# ===----------------------------------------------------------------------=== #
# Public API
# ===----------------------------------------------------------------------=== #


def psi(x: Float64) -> SFSResult:
    """Digamma function psi(x) = d/dx ln(Gamma(x)) for x > 0.

    Args:
        x: Argument, must be positive (x > 0).

    Returns:
        SFSResult with psi(x) and error estimate.
    """
    var result = SFSResult()
    if x <= 0.0:
        result.errno = MSL_EDOM
        return result^
    result.val = _psi_x(x)
    result.err = MSL_DBL_EPSILON * abs(result.val)
    return result^


def psi_n(n: Int, x: Float64) -> SFSResult:
    """Polygamma function psi^(n)(x) via recurrence from psi.

    For n == 0 this is just psi(x). For n >= 1 it uses the
    recurrence psi^(n)(x) = psi^(n)(x+1) - (-1)^n * n! / x^(n+1).

    Args:
        n: Order (n >= 0).
        x: Argument, must be positive.

    Returns:
        SFSResult with psi^(n)(x) and error estimate.
    """
    var result = SFSResult()
    if x <= 0.0 or n < 0:
        result.errno = MSL_EDOM
        return result^
    if n == 0:
        return psi(x)

    # Shift argument up until x_shift > 20, accumulate recurrence
    var x_shift = x
    var recurrence: Float64 = 0.0
    var sign: Float64 = 1.0 if n % 2 == 1 else -1.0

    var nfact: Float64 = 1.0
    for k in range(1, n + 1):
        nfact *= Float64(k)

    while x_shift < 20.0:
        recurrence += sign * nfact / (x_shift ** Float64(n + 1))
        x_shift += 1.0

    # Asymptotic: psi^(n)(x) ~ (-1)^(n+1) * (n-1)!/x^n * (1 + n/(2x) + ...)
    var w = 1.0 / (x_shift * x_shift)
    var sum_asym = (
        nfact
        / (x_shift ** Float64(n))
        * (
            0.5
            + x_shift
            * (
                1.0 / Float64(n)
                - w / 6.0 * (1.0 - w / 5.0 * (1.0 - 7.0 * w / 14.0))
            )
        )
    )
    result.val = sum_asym + recurrence
    result.err = MSL_DBL_EPSILON * abs(result.val)
    return result^
