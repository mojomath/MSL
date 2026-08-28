# SPDX-License-Identifier: GPL-3.0-or-later

# ===----------------------------------------------------------------------=== #
# MSL (Mojo Scientific Library)
#
# Derived from GNU Scientific Library (GSL)
# Original file: specfunc/erfc.c
#
# Original authors:
# Copyright (C) 1996-2003 Gerard Jungman
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
Error functions.


"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.collections import InlineArray
from std.math import (
    abs,
    exp,
    log,
)

# ===----------------------------------------------------------------------=== #
# MSL
# ===----------------------------------------------------------------------=== #
from msl.core.const import (
    MSL_DBL_EPSILON,
    MSL_SQRT2,
    MSL_SQRTPI,
)
from msl.sf.result import SFSResult


# ===----------------------------------------------------------------------=== #
# Chebyshev series evaluation
# ===----------------------------------------------------------------------=== #


def cheb_eval[
    N: Int
](
    c: InlineArray[Float64, N], order: Int, a: Float64, b: Float64, x: Float64
) -> SFSResult:
    var d: Float64 = 0.0
    var dd: Float64 = 0.0

    var y = (2.0 * x - a - b) / (b - a)
    var y2 = 2.0 * y

    var j = order
    while j >= 1:
        var temp = d
        d = y2 * d - dd + c[j]
        dd = temp
        j -= 1

    var result = SFSResult()
    result.val = y * d - dd + 0.5 * c[0]
    result.err = MSL_DBL_EPSILON * abs(result.val) + abs(c[order])
    return result^


# ===----------------------------------------------------------------------=== #
# erfc8: asymptotic expansion for large x (8 < x < 100)
# ===----------------------------------------------------------------------=== #


def erfc8_sum(x: Float64) -> Float64:
    var erfc8_P: InlineArray[Float64, 6] = [
        2.97886562639399288862,
        7.409740605964741794425,
        6.1602098531096305440906,
        5.019049726784267463450058,
        1.275366644729965952479585264,
        0.5641895835477550741253201704,
    ]
    var erfc8_Q: InlineArray[Float64, 7] = [
        3.3690752069827527677,
        9.608965327192787870698,
        17.08144074746600431571095,
        12.0489519278551290360340491,
        9.396034016235054150430579648,
        2.260528520767326969591866945,
        1.0,
    ]

    var num = erfc8_P[5]
    var i = 4
    while i >= 0:
        num = x * num + erfc8_P[i]
        i -= 1

    var den = erfc8_Q[6]
    i = 5
    while i >= 0:
        den = x * den + erfc8_Q[i]
        i -= 1

    return num / den


def erfc8(x: Float64) -> Float64:
    var e = erfc8_sum(x)
    return e * exp(-x * x)


# ===----------------------------------------------------------------------=== #
# erf series expansion for |x| < 1
# ===----------------------------------------------------------------------=== #


def erf_series(x: Float64) -> SFSResult:
    var coef = x
    var e = coef
    var delta: Float64 = 0.0

    var k = 1
    while k < 30:
        coef = -x * x / Float64(k)
        delta = coef / (2.0 * Float64(k) + 1.0)
        e += delta
        k += 1

    var result = SFSResult()
    result.val = 2.0 / MSL_SQRTPI * e
    result.err = 2.0 / MSL_SQRTPI * (abs(delta) + MSL_DBL_EPSILON)
    return result^


# ===----------------------------------------------------------------------=== #
# Error function erf(x)
# ===----------------------------------------------------------------------=== #


def erf(x: Float64) -> SFSResult:
    if abs(x) < 1.0:
        return erf_series(x)
    else:
        var result_erfc = erfc(x)
        var result = SFSResult()
        result.val = 1.0 - result_erfc.val
        result.err = result_erfc.err + 2.0 * MSL_DBL_EPSILON * abs(result.val)
        return result^


# ===----------------------------------------------------------------------=== #
# Complementary error function erfc(x)
# ===----------------------------------------------------------------------=== #


def erfc(x: Float64) -> SFSResult:
    var result = SFSResult()

    if x < 0.0:
        var erfc_pos = erfc(-x)
        result.val = 2.0 - erfc_pos.val
        result.err = erfc_pos.err + 2.0 * MSL_DBL_EPSILON * abs(result.val)
        return result^

    if x < 1.0:
        var et_xlt1: InlineArray[Float64, 21] = [
            1.06073416421769980345174155056,
            -0.42582445804381043569204735291,
            0.04955262679620434040357683080,
            0.00449293488768382749558001242,
            -0.00129194104658496953494224761,
            -0.00001836389292149396270416979,
            0.00002211114704099526291538556,
            -5.23337485234257134673693179020e-7,
            -1.762747542774542660477945670e-7,
            3.190822782374654761984499495e-8,
            -5.068297400714582694180541675e-9,
            6.375789646498584765053532028e-9,
            -4.687656084110757156942656933e-9,
            1.050948467115521956210555952e-9,
            1.193584501822410572511108519e-9,
            -8.401541651479210250766091810e-10,
            -3.252197108330553889166550877e-10,
            3.890106529690055727242933584e-10,
            -1.418272597129287798518395778e-10,
            -4.432538519890313073697460291e-11,
            4.522355270409269485131849038e-11,
        ]
        var t = 2.0 * x - 1.0
        var c = cheb_eval(et_xlt1, 20, -1.0, 1.0, t)
        result.val = c.val * exp(-x * x)
        result.err = c.err + MSL_DBL_EPSILON * abs(result.val)
        return result^

    if x < 8.0:
        var et_xgt1: InlineArray[Float64, 17] = [
            2.96978671319817e-16,
            3.62233517000948e-14,
            2.64316767841567e-12,
            1.71578146840131e-10,
            9.16595344644874e-09,
            4.18467273979012e-07,
            1.65239998842208e-05,
            5.692743634145e-04,
            1.66985726753879e-02,
            4.15894254202913e-01,
            8.54042631170831e-01,
            1.40544340996924e00,
            1.84206010699563e00,
            1.00000000000000e00,
            3.77383527859054e-01,
            9.86218717458347e-02,
            1.68450166415940e-02,
        ]
        var t = (x - 4.5) / 3.5
        var c = cheb_eval(et_xgt1, 16, -1.0, 1.0, t)
        result.val = c.val * exp(-x * x)
        result.err = c.err + MSL_DBL_EPSILON * abs(result.val)
        return result^

    # x >= 8.0
    result.val = erfc8(x)
    result.err = 2.0 * MSL_DBL_EPSILON * abs(result.val)
    return result^


# ===----------------------------------------------------------------------=== #
# Log complementary error function
# ===----------------------------------------------------------------------=== #


def log_erfc(x: Float64) -> SFSResult:
    var result = SFSResult()

    if x < 0.0:
        var log_erfc_pos = log_erfc(-x)
        var v = log(2.0 - exp(log_erfc_pos.val))
        result.val = v
        result.err = 2.0 * MSL_DBL_EPSILON * abs(v)
        return result^

    if x < 1.0:
        var et_xlt1: InlineArray[Float64, 21] = [
            1.06073416421769980345174155056,
            -0.42582445804381043569204735291,
            0.04955262679620434040357683080,
            0.00449293488768382749558001242,
            -0.00129194104658496953494224761,
            -0.00001836389292149396270416979,
            0.00002211114704099526291538556,
            -5.23337485234257134673693179020e-7,
            -1.762747542774542660477945670e-7,
            3.190822782374654761984499495e-8,
            -5.068297400714582694180541675e-9,
            6.375789646498584765053532028e-9,
            -4.687656084110757156942656933e-9,
            1.050948467115521956210555952e-9,
            1.193584501822410572511108519e-9,
            -8.401541651479210250766091810e-10,
            -3.252197108330553889166550877e-10,
            3.890106529690055727242933584e-10,
            -1.418272597129287798518395778e-10,
            -4.432538519890313073697460291e-11,
            4.522355270409269485131849038e-11,
        ]
        var t = 2.0 * x - 1.0
        var c = cheb_eval(et_xlt1, 20, -1.0, 1.0, t)
        result.val = log(c.val) - x * x
        result.err = 2.0 * MSL_DBL_EPSILON * abs(result.val)
        return result^

    if x < 8.0:
        var et_xgt1: InlineArray[Float64, 17] = [
            2.96978671319817e-16,
            3.62233517000948e-14,
            2.64316767841567e-12,
            1.71578146840131e-10,
            9.16595344644874e-09,
            4.18467273979012e-07,
            1.65239998842208e-05,
            5.692743634145e-04,
            1.66985726753879e-02,
            4.15894254202913e-01,
            8.54042631170831e-01,
            1.40544340996924e00,
            1.84206010699563e00,
            1.00000000000000e00,
            3.77383527859054e-01,
            9.86218717458347e-02,
            1.68450166415940e-02,
        ]
        var t = (x - 4.5) / 3.5
        var c = cheb_eval(et_xgt1, 16, -1.0, 1.0, t)
        result.val = log(c.val) - x * x
        result.err = 2.0 * MSL_DBL_EPSILON * abs(result.val)
        return result^

    # x >= 8.0
    var e = erfc8_sum(x)
    result.val = log(e) - x * x
    result.err = 2.0 * MSL_DBL_EPSILON * abs(result.val)
    return result^


# ===----------------------------------------------------------------------=== #
# Probability functions: Z(x) = (1/sqrt(2pi)) * exp(-x^2/2)
# ===----------------------------------------------------------------------=== #


def erf_Z(x: Float64) -> SFSResult:
    var result = SFSResult()
    var ex2 = exp(-x * x / 2.0)
    result.val = ex2 / (MSL_SQRT2 * MSL_SQRTPI)
    result.err = abs(
        x * result.val
    ) * MSL_DBL_EPSILON + 2.0 * MSL_DBL_EPSILON * abs(result.val)
    return result^


# ===----------------------------------------------------------------------=== #
# Q(x) = 0.5 * erfc(x/sqrt(2))
# ===----------------------------------------------------------------------=== #


def erf_Q(x: Float64) -> SFSResult:
    var result_erfc = erfc(x / MSL_SQRT2)
    var result = SFSResult()
    result.val = 0.5 * result_erfc.val
    result.err = 0.5 * result_erfc.err + 2.0 * MSL_DBL_EPSILON * abs(result.val)
    return result^


# ===----------------------------------------------------------------------=== #
# Hazard function H(x) = Z(x) / Q(x)
# ===----------------------------------------------------------------------=== #


def hazard(x: Float64) -> SFSResult:
    var result = SFSResult()

    if x < 25.0:
        var result_ln_erfc = log_erfc(x / MSL_SQRT2)
        var lnc = -0.22579135264472743236
        var arg = lnc - 0.5 * x * x - result_ln_erfc.val
        result.val = exp(arg)
        result.err = 3.0 * (1.0 + abs(x)) * MSL_DBL_EPSILON * abs(result.val)
        result.err += abs(result_ln_erfc.err * result.val)
        return result^
    else:
        var ix2 = 1.0 / (x * x)
        var corrB = 1.0 - 9.0 * ix2 * (1.0 - 11.0 * ix2)
        var corrM = 1.0 - 5.0 * ix2 * (1.0 - 7.0 * ix2 * corrB)
        var corrT = 1.0 - ix2 * (1.0 - 3.0 * ix2 * corrM)
        result.val = x / corrT
        result.err = 2.0 * MSL_DBL_EPSILON * abs(result.val)
        return result^
