# SPDX-License-Identifier: GPL-3.0-or-later

# ===----------------------------------------------------------------------=== #
# MSL (Mojo Scientific Library)
#
# Derived from GNU Scientific Library (GSL)
# Original file: integration/qk.c, integration/err.c
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
Generic n-point Gauss-Kronrod quadrature and error rescaling.
"""

from msl.core.const import MSL_DBL_EPSILON, MSL_DBL_MIN
from std.math import abs, pow

from .workspace import QKResult


def _rescale_error(
    err: Float64, result_abs: Float64, result_asc: Float64
) -> Float64:
    """Rescale the error estimate following GSL convention."""
    var e = abs(err)

    if result_asc != 0.0 and e != 0.0:
        var scale = pow(200.0 * e / result_asc, 1.5)
        if scale < 1.0:
            e = result_asc * scale
        else:
            e = result_asc

    if result_abs > MSL_DBL_MIN / (50.0 * MSL_DBL_EPSILON):
        var min_err = 50.0 * MSL_DBL_EPSILON * result_abs
        if min_err > e:
            e = min_err

    return e


def _qk_generic[
    fn_: def(Float64) -> Float64,
](
    n: Int,
    xgk: InlineArray[Float64, _],
    wg: InlineArray[Float64, _],
    wgk: InlineArray[Float64, _],
    mut fv1: InlineArray[Float64, _],
    mut fv2: InlineArray[Float64, _],
    a: Float64,
    b: Float64,
) -> QKResult:
    """Generic n-point Gauss-Kronrod quadrature.

    Parameters:
        f: Function to integrate

    Args:
        n: Number of Kronrod points (half + 1 for symmetric rule)
        xgk: Kronrod abscissae (size n)
        wg: Gauss weights (size n//2 or (n-1)//2 + 1)
        wgk: Kronrod weights (size n)
        fv1: Workspace for function values (size n)
        fv2: Workspace for function values (size n)
        a: Lower limit
        b: Upper limit

    Returns:
        QKResult with result_kronrod, abserr, resabs, resasc
    """
    var center = 0.5 * (a + b)
    var half_length = 0.5 * (b - a)
    var abs_half_length = abs(half_length)
    var f_center = fn_(center)

    var result_gauss: Float64 = 0.0
    var result_kronrod = f_center * wgk[n - 1]
    var result_abs = abs(result_kronrod)
    var result_asc: Float64

    if n % 2 == 0:
        result_gauss = f_center * wg[n // 2 - 1]

    for j in range((n - 1) // 2):
        var jtw = j * 2 + 1
        var abscissa = half_length * xgk[jtw]
        var fval1 = fn_(center - abscissa)
        var fval2 = fn_(center + abscissa)
        var fsum = fval1 + fval2
        fv1[jtw] = fval1
        fv2[jtw] = fval2
        result_gauss += wg[j] * fsum
        result_kronrod += wgk[jtw] * fsum
        result_abs += wgk[jtw] * (abs(fval1) + abs(fval2))

    for j in range(n // 2):
        var jtwm1 = j * 2
        var abscissa = half_length * xgk[jtwm1]
        var fval1 = fn_(center - abscissa)
        var fval2 = fn_(center + abscissa)
        fv1[jtwm1] = fval1
        fv2[jtwm1] = fval2
        result_kronrod += wgk[jtwm1] * (fval1 + fval2)
        result_abs += wgk[jtwm1] * (abs(fval1) + abs(fval2))

    var mean = result_kronrod * 0.5
    result_asc = wgk[n - 1] * abs(f_center - mean)

    for j in range(n - 1):
        result_asc += wgk[j] * (abs(fv1[j] - mean) + abs(fv2[j] - mean))

    var err = (result_kronrod - result_gauss) * half_length
    result_kronrod *= half_length
    result_abs *= abs_half_length
    result_asc *= abs_half_length

    var abserr = _rescale_error(err, result_abs, result_asc)

    var res = QKResult()
    res.result = result_kronrod
    res.abserr = abserr
    res.resabs = result_abs
    res.resasc = result_asc
    return res^
