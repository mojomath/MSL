# SPDX-License-Identifier: GPL-3.0-or-later

# ===----------------------------------------------------------------------=== #
# MSL (Mojo Scientific Library)
#
# Derived from GNU Scientific Library (GSL)
# Original file: integration/qag.c
#
# Original authors:
# Copyright (C) 1996-2007 Brian Gough
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
QAG — General adaptive Gauss-Kronrod integration.

Bisects the interval with the largest error estimate until the global
error is within tolerance, or the subinterval limit is reached.
"""

from std.math import abs

from msl.core.const import MSL_DBL_EPSILON, MSL_DBL_MIN
from msl.core.errno import MSL_EDOM, MSL_EMAXITER, MSL_EROUND, MSL_ESING
from msl.sf.result import SFSResult
from .workspace import IntegrationWorkspace, IntegrationResult, QKResult
from .qk15 import qk15
from .qk21 import qk21
from .qk31 import qk31
from .qk41 import qk41
from .qk51 import qk51
from .qk61 import qk61

comptime MSL_INTEG_GAUSS15: Int = 1
comptime MSL_INTEG_GAUSS21: Int = 2
comptime MSL_INTEG_GAUSS31: Int = 3
comptime MSL_INTEG_GAUSS41: Int = 4
comptime MSL_INTEG_GAUSS51: Int = 5
comptime MSL_INTEG_GAUSS61: Int = 6


def _subinterval_too_small(a1: Float64, a2: Float64, b2: Float64) -> Bool:
    var tmp = (1.0 + 100.0 * MSL_DBL_EPSILON) * (abs(a2) + 1000.0 * MSL_DBL_MIN)
    return abs(a1) <= tmp and abs(b2) <= tmp


def _apply_rule[
    integrand: def(Float64) capturing -> Float64
](a: Float64, b: Float64, key: Int) -> QKResult:
    if key == MSL_INTEG_GAUSS21:
        return qk21[integrand](a, b)
    elif key == MSL_INTEG_GAUSS31:
        return qk31[integrand](a, b)
    elif key == MSL_INTEG_GAUSS41:
        return qk41[integrand](a, b)
    elif key == MSL_INTEG_GAUSS51:
        return qk51[integrand](a, b)
    elif key == MSL_INTEG_GAUSS61:
        return qk61[integrand](a, b)
    else:
        return qk15[integrand](a, b)


def qag[
    integrand: def(Float64) capturing -> Float64
](
    a: Float64,
    b: Float64,
    epsabs: Float64,
    epsrel: Float64,
    limit: Int = 50,
    key: Int = MSL_INTEG_GAUSS21,
) -> IntegrationResult:
    """Adaptive Gauss-Kronrod integration of integrand over [a, b].

    Parameters:
        integrand: Scalar function to integrate.

    Args:
        a: Lower limit.
        b: Upper limit.
        epsabs: Absolute error tolerance.
        epsrel: Relative error tolerance.
        limit: Maximum number of subintervals (default 50).
        key: Quadrature rule - one of MSL_INTEG_GAUSS{15,21,31,41,51,61}.

    Returns:
        IntegrationResult with val (integral estimate) and err (error bound).
    """
    var result = IntegrationResult()

    var r0 = _apply_rule[integrand](a, b, key=key)
    var area = r0.result
    var errsum = r0.abserr
    var resabs0 = r0.resabs
    var resasc0 = r0.resasc

    var tolerance = max(epsabs, epsrel * abs(area))
    var round_off = 50.0 * MSL_DBL_EPSILON * resabs0

    if errsum <= round_off and errsum > tolerance:
        result.val = area
        result.err = errsum
        return result^

    if (errsum <= tolerance and errsum != resasc0) or errsum == 0.0:
        result.val = area
        result.err = errsum
        return result^

    var ws = IntegrationWorkspace(limit)
    ws.initialise(a, b, area, errsum)

    var roundoff_type1: Int = 0
    var roundoff_type2: Int = 0
    var error_type: Int = 0

    for iteration in range(1, limit):
        var a_i: Float64
        var b_i: Float64
        var r_i: Float64
        var e_i: Float64
        a_i, b_i, r_i, e_i = ws.retrieve()

        var mid = 0.5 * (a_i + b_i)
        var r1 = _apply_rule[integrand](a_i, mid, key=key)
        var r2 = _apply_rule[integrand](mid, b_i, key=key)

        var area12 = r1.result + r2.result
        var error12 = r1.abserr + r2.abserr

        errsum += error12 - e_i
        area += area12 - r_i

        if r1.resasc != r1.abserr and r2.resasc != r2.abserr:
            var delta = r_i - area12
            if abs(delta) <= 1.0e-5 * abs(area12) and error12 >= 0.99 * e_i:
                roundoff_type1 += 1
            if iteration >= 10 and error12 > e_i:
                roundoff_type2 += 1

        tolerance = max(epsabs, epsrel * abs(area))

        if errsum > tolerance:
            if roundoff_type1 >= 6 or roundoff_type2 >= 20:
                error_type = 2
            if _subinterval_too_small(a_i, mid, b_i):
                error_type = 3

        ws.update(
            a_i, mid, r1.result, r1.abserr, mid, b_i, r2.result, r2.abserr
        )

        if error_type != 0:
            break

        if errsum <= tolerance:
            break

    result.val = ws.sum_results()
    result.err = errsum
    return result^
