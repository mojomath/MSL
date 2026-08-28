# SPDX-License-Identifier: GPL-3.0-or-later

# ===----------------------------------------------------------------------=== #
# MSL (Mojo Scientific Library)
#
# Derived from GNU Scientific Library (GSL)
# Original file: integration/qk15.c
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
15-point Gauss-Kronrod quadrature rule.
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.math import (
    abs,
    pow,
)

# ===----------------------------------------------------------------------=== #
# MSL
# ===----------------------------------------------------------------------=== #
from msl.core.const import (
    MSL_DBL_EPSILON,
    MSL_DBL_MIN,
)
from msl.integration.workspace import QKResult


comptime _xgk: Array[Float64, 8] = [
    0.991455371120812639206854697526329,
    0.949107912342758524526189684047851,
    0.864864423359769072789712788640926,
    0.741531185599394439863864773280788,
    0.586087235467691130294144838258730,
    0.405845151377397166906606412076961,
    0.207784955007898467600689403773245,
    0.000000000000000000000000000000000,
]

comptime _wg: Array[Float64, 4] = [
    0.129484966168869693270611432679082,
    0.279705391489276667901467771423780,
    0.381830050505118944950369775488975,
    0.417959183673469387755102040816327,
]

comptime _wgk: Array[Float64, 8] = [
    0.022935322010529224963732008058970,
    0.063092092629978553290700663189204,
    0.104790010322250183839876322541518,
    0.140653259715525918745189590510238,
    0.169004726639267902826583426598550,
    0.190350578064785409913256402421014,
    0.204432940075298892414161999234649,
    0.209482141084727828012999174891714,
]


def _rescale_error(
    err: Float64, result_abs: Float64, result_asc: Float64
) -> Float64:
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


def qk15[
    fn_: def(Float64) capturing -> Float64
](a: Float64, b: Float64) -> QKResult:
    var n = 8

    var fv1_0: Float64 = 0.0
    var fv1_1: Float64 = 0.0
    var fv1_2: Float64 = 0.0
    var fv1_3: Float64 = 0.0
    var fv1_4: Float64 = 0.0
    var fv1_5: Float64 = 0.0
    var fv1_6: Float64 = 0.0
    var fv1_7: Float64 = 0.0

    var fv2_0: Float64 = 0.0
    var fv2_1: Float64 = 0.0
    var fv2_2: Float64 = 0.0
    var fv2_3: Float64 = 0.0
    var fv2_4: Float64 = 0.0
    var fv2_5: Float64 = 0.0
    var fv2_6: Float64 = 0.0
    var fv2_7: Float64 = 0.0

    var center = 0.5 * (a + b)
    var half_length = 0.5 * (b - a)
    var abs_half_length = abs(half_length)
    var f_center = fn_(center)

    var result_gauss: Float64 = 0.0
    var result_kronrod = f_center * _wgk[n - 1]
    var result_abs = abs(result_kronrod)
    var result_asc: Float64

    if n % 2 == 0:
        result_gauss = f_center * _wg[n // 2 - 1]

    for j in range((n - 1) // 2):
        var jtw = j * 2 + 1
        var abscissa = half_length * _xgk[jtw]
        var fval1 = fn_(center - abscissa)
        var fval2 = fn_(center + abscissa)
        var fsum = fval1 + fval2

        if jtw == 0:
            fv1_0 = fval1
            fv2_0 = fval2
        elif jtw == 1:
            fv1_1 = fval1
            fv2_1 = fval2
        elif jtw == 2:
            fv1_2 = fval1
            fv2_2 = fval2
        elif jtw == 3:
            fv1_3 = fval1
            fv2_3 = fval2
        elif jtw == 4:
            fv1_4 = fval1
            fv2_4 = fval2
        elif jtw == 5:
            fv1_5 = fval1
            fv2_5 = fval2
        elif jtw == 6:
            fv1_6 = fval1
            fv2_6 = fval2
        elif jtw == 7:
            fv1_7 = fval1
            fv2_7 = fval2

        result_gauss += _wg[j] * fsum
        result_kronrod += _wgk[jtw] * fsum
        result_abs += _wgk[jtw] * (abs(fval1) + abs(fval2))

    for j in range(n // 2):
        var jtwm1 = j * 2
        var abscissa = half_length * _xgk[jtwm1]
        var fval1 = fn_(center - abscissa)
        var fval2 = fn_(center + abscissa)

        if jtwm1 == 0:
            fv1_0 = fval1
            fv2_0 = fval2
        elif jtwm1 == 1:
            fv1_1 = fval1
            fv2_1 = fval2
        elif jtwm1 == 2:
            fv1_2 = fval1
            fv2_2 = fval2
        elif jtwm1 == 3:
            fv1_3 = fval1
            fv2_3 = fval2
        elif jtwm1 == 4:
            fv1_4 = fval1
            fv2_4 = fval2
        elif jtwm1 == 5:
            fv1_5 = fval1
            fv2_5 = fval2
        elif jtwm1 == 6:
            fv1_6 = fval1
            fv2_6 = fval2

        result_kronrod += _wgk[jtwm1] * (fval1 + fval2)
        result_abs += _wgk[jtwm1] * (abs(fval1) + abs(fval2))

    var mean = result_kronrod * 0.5
    result_asc = _wgk[n - 1] * abs(f_center - mean)

    result_asc += _wgk[0] * (abs(fv1_0 - mean) + abs(fv2_0 - mean))
    result_asc += _wgk[1] * (abs(fv1_1 - mean) + abs(fv2_1 - mean))
    result_asc += _wgk[2] * (abs(fv1_2 - mean) + abs(fv2_2 - mean))
    result_asc += _wgk[3] * (abs(fv1_3 - mean) + abs(fv2_3 - mean))
    result_asc += _wgk[4] * (abs(fv1_4 - mean) + abs(fv2_4 - mean))
    result_asc += _wgk[5] * (abs(fv1_5 - mean) + abs(fv2_5 - mean))
    result_asc += _wgk[6] * (abs(fv1_6 - mean) + abs(fv2_6 - mean))
    result_asc += _wgk[7] * (abs(fv1_7 - mean) + abs(fv2_7 - mean))

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
