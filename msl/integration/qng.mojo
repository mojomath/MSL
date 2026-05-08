# SPDX-License-Identifier: GPL-3.0-or-later

# ===----------------------------------------------------------------------=== #
# MSL (Mojo Scientific Library)
#
# Derived from GNU Scientific Library (GSL)
# Original file: integration/qng.c, integration/qng.h
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
QNG - Non-adaptive Gauss-Kronrod integration.
"""

from msl.core.const import MSL_DBL_EPSILON, MSL_SQRT_DBL_EPSILON
from std.math import abs, sqrt

from .workspace import IntegrationResult


comptime x1: InlineArray[Float64, 5] = [
    0.973906528517171720077964012084452,
    0.865063366688984510732096688423493,
    0.679409568299024406234327365114874,
    0.433395394129247190799265943165784,
    0.148874338981631210884826001129720,
]

comptime w10: InlineArray[Float64, 5] = [
    0.066671344308688137593568809893332,
    0.149451349150580593145776339657697,
    0.219086362515982043995534934228163,
    0.269266719309996355091226921569469,
    0.295524224714752870173892994651338,
]

comptime x2: InlineArray[Float64, 5] = [
    0.995657163025808080735527280689003,
    0.930157491355708226001207180059508,
    0.780817726586416897063717578345042,
    0.562757134668604683339000099272694,
    0.294392862701460198131126603103866,
]

comptime w21a: InlineArray[Float64, 5] = [
    0.032558162307964727478818972459390,
    0.075039674810919952767043140916190,
    0.109387158802297641899210590325805,
    0.134709217311473325928054001771707,
    0.147739104901338491374841515972068,
]

comptime w21b: InlineArray[Float64, 6] = [
    0.011694638867371874278064396062192,
    0.054755896574351996031381300244580,
    0.093125454583697605535065465083366,
    0.123491976262065851077958109831074,
    0.142775938577060080797094273138717,
    0.149445554002916905664936468389821,
]


def qng_integrate[
    fn_: def(Float64) capturing -> Float64
](
    a: Float64, b: Float64, epsabs: Float64, epsrel: Float64
) -> IntegrationResult:
    var fv1 = InlineArray[Float64, 5](uninitialized=True)
    var fv2 = InlineArray[Float64, 5](uninitialized=True)
    var fv3 = InlineArray[Float64, 5](uninitialized=True)
    var fv4 = InlineArray[Float64, 5](uninitialized=True)
    var savfun = InlineArray[Float64, 21](uninitialized=True)

    var res10: Float64 = 0.0
    var res21: Float64 = w21b[5] * fn_(0.5 * (b + a))
    var resabs: Float64 = w21b[5] * abs(fn_(0.5 * (b + a)))
    var res43: Float64
    var res87: Float64

    var half_length = 0.5 * (b - a)
    var abs_half_length = abs(half_length)
    var center = 0.5 * (b + a)
    var f_center = fn_(center)

    if epsabs <= 0.0 and (epsrel < 50.0 * MSL_DBL_EPSILON or epsrel < 0.5e-28):
        var res = IntegrationResult()
        return res^

    var k = 0
    while k < 5:
        var abscissa = half_length * x1[k]
        var fval1 = fn_(center + abscissa)
        var fval2 = fn_(center - abscissa)
        var fval = fval1 + fval2
        res10 += w10[k] * fval
        res21 += w21a[k] * fval
        resabs += w21a[k] * (abs(fval1) + abs(fval2))
        savfun[k] = fval
        fv1[k] = fval1
        fv2[k] = fval2
        k += 1

    k = 0
    while k < 5:
        var abscissa = half_length * x2[k]
        var fval1 = fn_(center + abscissa)
        var fval2 = fn_(center - abscissa)
        var fval = fval1 + fval2
        res21 += w21b[k] * fval
        resabs += w21b[k] * (abs(fval1) + abs(fval2))
        savfun[k + 5] = fval
        fv3[k] = fval1
        fv4[k] = fval2
        k += 1

    resabs *= abs_half_length

    var mean = 0.5 * res21
    var resasc = w21b[5] * abs(f_center - mean)

    k = 0
    while k < 5:
        resasc += w21a[k] * (abs(fv1[k] - mean) + abs(fv2[k] - mean))
        k += 1

    k = 0
    while k < 5:
        resasc += w21b[k] * (abs(fv3[k] - mean) + abs(fv4[k] - mean))
        k += 1

    resasc *= abs_half_length

    var result_kronrod = res21 * half_length
    var err = abs(res21 - res10) * half_length

    if resasc != 0.0:
        err = err * resasc / (resabs + resasc)

    if resabs >= 1e-5 and resasc >= 1e-5:
        var aa = (resasc / resabs) * (resasc / resabs)
        if aa < 1.0:
            err = err * sqrt(aa)
            err = err / MSL_DBL_EPSILON

    var abserr = err

    var res = IntegrationResult()
    res.val = result_kronrod
    res.err = abserr
    return res^
