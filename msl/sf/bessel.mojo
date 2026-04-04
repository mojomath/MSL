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
Bessel functions.

This is a direct port of the GSL special functions implementation.
"""

# TODO: Cross check this with CEPHES library implementations for accuracy.

from msl.core.const import (
    MSL_DBL_EPSILON,
    MSL_SQRT_DBL_EPSILON,
    MSL_PI,
    MSL_PI_2,
    MSL_PI_4,
)
from msl.sf.result import SFSResult
from std.math import sqrt, exp, cos, sin, abs, log
from std.collections import InlineArray


comptime SFPrecisionDouble: Int = 0
comptime SFPrecisionSingle: Int = 1
comptime SFPrecisionApprox: Int = 2


# ===----------------------------------------------------------------------=== #
# Chebyshev series evaluation
# ===----------------------------------------------------------------------=== #


# TODO: I think I can move order, a and b to comptime and write comptime for loop!
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
# Bessel J0
# ===----------------------------------------------------------------------=== #


def bessel_J0(x: Float64) -> SFSResult:
    var y = abs(x)

    if y < 2.0 * MSL_SQRT_DBL_EPSILON:
        var result = SFSResult()
        result.val = 1.0
        result.err = y * y
        return result
    elif y <= 4.0:
        var bj0_c: InlineArray[Float64, 13] = [
            0.100254161968939137,
            -0.665223007764405132,
            0.248983703498281314,
            -0.0332527231700357697,
            0.0023114179304694015,
            -0.0000991127741995080,
            0.0000028916708643998,
            -0.0000000612108586630,
            0.0000000009838650793,
            -0.0000000000124235515,
            0.0000000000001265433,
            -0.0000000000000010619,
            0.0000000000000000074,
        ]
        return cheb_eval(bj0_c, 12, -1.0, 1.0, 0.125 * y * y - 1.0)
    else:
        var z = 32.0 / (y * y) - 1.0
        var sqrty = sqrt(y)

        var bm0_c: InlineArray[Float64, 21] = [
            0.09284961637381644,
            -0.00142987707403484,
            0.00002830579271257,
            -0.00000143300611424,
            0.00000012028628046,
            -0.00000001397113013,
            0.00000000204076188,
            -0.00000000035399669,
            0.00000000007024759,
            -0.00000000001554107,
            0.00000000000376226,
            -0.00000000000098282,
            0.00000000000027408,
            -0.00000000000008091,
            0.00000000000002511,
            -0.00000000000000814,
            0.00000000000000275,
            -0.00000000000000096,
            0.00000000000000034,
            -0.00000000000000012,
            0.00000000000000004,
        ]
        var ca = cheb_eval(bm0_c, 20, -1.0, 1.0, z)

        var bth0_c: InlineArray[Float64, 24] = [
            -0.24639163774300119,
            0.001737098307508963,
            -0.000062183633402968,
            0.000004368050165742,
            -0.000000456093019869,
            0.000000062197400101,
            -0.000000010300442889,
            0.000000001979526776,
            -0.000000000428198396,
            0.000000000102035840,
            -0.000000000026363898,
            0.000000000007297935,
            -0.000000000002144188,
            0.000000000000663693,
            -0.000000000000215126,
            0.000000000000072659,
            -0.000000000000025465,
            0.000000000000009229,
            -0.000000000000003448,
            0.000000000000001325,
            -0.000000000000000522,
            0.000000000000000210,
            -0.000000000000000087,
            0.000000000000000036,
        ]
        var ct = cheb_eval(bth0_c, 23, -1.0, 1.0, z)

        var theta = ct.val / y
        var cp_val = cos(y - MSL_PI_4 + theta)
        var cp_err = MSL_DBL_EPSILON * abs(cp_val)

        var ampl = (0.75 + ca.val) / sqrty

        var result = SFSResult()
        result.val = ampl * cp_val
        result.err = abs(cp_val) * ca.err / sqrty + abs(ampl) * cp_err
        result.err += MSL_DBL_EPSILON * abs(result.val)
        return result^


# ===----------------------------------------------------------------------=== #
# Bessel J1
# ===----------------------------------------------------------------------=== #


def bessel_J1(x: Float64) -> SFSResult:
    var y = abs(x)

    if y < 2.0 * MSL_SQRT_DBL_EPSILON:
        var result = SFSResult()
        result.val = 0.5 * x
        result.err = y * y
        return result
    elif y <= 4.0:
        var bj1_c: InlineArray[Float64, 13] = [
            0.0149447472378184,
            -0.425408808762136,
            0.254653877192681,
            -0.0542093523932086,
            0.00580319435876507,
            -0.000351376204508767,
            0.0000145252747076479,
            -0.000000445259377622,
            0.0000000106956683789,
            -0.000000000205635845,
            0.0000000000032395659,
            -0.0000000000000421514,
            0.000000000000000459,
        ]
        var z = 0.125 * y * y - 1.0
        var result_c = cheb_eval(bj1_c, 12, -1.0, 1.0, z)

        var result = SFSResult()
        if x > 0.0:
            result.val = y * result_c.val
        else:
            result.val = -y * result_c.val
        result.err = y * result_c.err
        result.err += MSL_DBL_EPSILON * abs(result.val)
        return result^
    else:
        var z = 32.0 / (y * y) - 1.0
        var sqrty = sqrt(y)
        var sign_x = 1.0 if x > 0.0 else -1.0

        var bm1_c: InlineArray[Float64, 21] = [
            0.1170304509391578,
            -0.0021846037119180,
            0.0000644125585904,
            -0.0000051198594778,
            0.0000005750282432,
            -0.0000000851897998,
            0.0000000154924449,
            -0.0000000032943850,
            0.0000000007901330,
            -0.0000000002089260,
            0.0000000000597373,
            -0.0000000000181477,
            0.0000000000057943,
            -0.0000000000019334,
            0.0000000000006704,
            -0.0000000000002402,
            0.0000000000000887,
            -0.0000000000000336,
            0.0000000000000130,
            -0.0000000000000051,
            0.0000000000000020,
        ]
        var ca = cheb_eval(bm1_c, 20, -1.0, 1.0, z)

        var bth1_c: InlineArray[Float64, 24] = [
            -0.0789742881163768,
            0.0020532235061588,
            -0.0001144014099977,
            0.0000098938054817,
            -0.0000012209770431,
            0.0000001863854547,
            -0.0000000340269543,
            0.0000000071604194,
            -0.0000000016934885,
            0.0000000004406101,
            -0.0000000001236186,
            0.0000000000368366,
            -0.0000000000115636,
            0.0000000000037983,
            -0.0000000000012947,
            0.0000000000004574,
            -0.0000000000001667,
            0.0000000000000625,
            -0.0000000000000240,
            0.0000000000000094,
            -0.0000000000000038,
            0.0000000000000015,
            -0.0000000000000006,
        ]
        var ct = cheb_eval(bth1_c, 23, -1.0, 1.0, z)

        var theta = ct.val / y
        var cp_val = cos(y - 3.0 * MSL_PI_4 + theta)
        var cp_err = MSL_DBL_EPSILON * abs(cp_val)

        var ampl = (0.375 + ca.val) / sqrty

        var result = SFSResult()
        result.val = sign_x * ampl * cp_val
        result.err = abs(ampl) * cp_err + abs(cp_val) * ca.err / sqrty
        result.err += MSL_DBL_EPSILON * abs(result.val)
        return result^


# ===----------------------------------------------------------------------=== #
# Bessel Y0
# ===----------------------------------------------------------------------=== #


def bessel_Y0(x: Float64) -> SFSResult:
    if x <= 0.0:
        var result = SFSResult()
        result.val = 0.0
        result.err = 0.0
        return result

    var y = x

    if y < 2.0 * MSL_SQRT_DBL_EPSILON:
        var result = SFSResult()
        result.val = -(2.0 / MSL_PI) * log(y)
        result.err = 1.0
        return result
    elif y <= 4.0:
        var by0_c: InlineArray[Float64, 13] = [
            -0.0248403262838782,
            -1.874611202953201,
            0.685402772065160,
            -0.0848890840618024,
            0.0063121433514150,
            -0.000305013247269,
            0.0000100105223278,
            -0.0000002473414171,
            0.0000000045420539,
            -0.0000000000659238,
            0.0000000000007784,
            -0.0000000000000076,
            0.0000000000000001,
        ]
        var z = 0.125 * y * y - 1.0
        var result_c = cheb_eval(by0_c, 12, -1.0, 1.0, z)

        var result = SFSResult()
        result.val = result_c.val + (2.0 / MSL_PI) * log(y) * bessel_J0(y).val
        result.err = (
            abs(result_c.err)
            + abs(2.0 / MSL_PI) * abs(log(y)) * bessel_J0(y).err
        )
        result.err += MSL_DBL_EPSILON * abs(result.val)
        return result^
    else:
        var z = 32.0 / (y * y) - 1.0
        var sqrty = sqrt(y)

        var bm0_c: InlineArray[Float64, 21] = [
            0.09284961637381644,
            -0.00142987707403484,
            0.00002830579271257,
            -0.00000143300611424,
            0.00000012028628046,
            -0.00000001397113013,
            0.00000000204076188,
            -0.00000000035399669,
            0.00000000007024759,
            -0.00000000001554107,
            0.00000000000376226,
            -0.00000000000098282,
            0.00000000000027408,
            -0.00000000000008091,
            0.00000000000002511,
            -0.00000000000000814,
            0.00000000000000275,
            -0.00000000000000096,
            0.00000000000000034,
            -0.00000000000000012,
            0.00000000000000004,
        ]
        var ca = cheb_eval(bm0_c, 20, -1.0, 1.0, z)

        var bth0_c: InlineArray[Float64, 24] = [
            -0.24639163774300119,
            0.001737098307508963,
            -0.000062183633402968,
            0.000004368050165742,
            -0.000000456093019869,
            0.000000062197400101,
            -0.000000010300442889,
            0.000000001979526776,
            -0.000000000428198396,
            0.000000000102035840,
            -0.000000000026363898,
            0.000000000007297935,
            -0.000000000002144188,
            0.000000000000663693,
            -0.000000000000215126,
            0.000000000000072659,
            -0.000000000000025465,
            0.000000000000009229,
            -0.000000000000003448,
            0.000000000000001325,
            -0.000000000000000522,
            0.000000000000000210,
            -0.000000000000000087,
            0.000000000000000036,
        ]
        var ct = cheb_eval(bth0_c, 23, -1.0, 1.0, z)

        var theta = ct.val / y
        var sp_val = sin(y - MSL_PI_4 + theta)
        var sp_err = MSL_DBL_EPSILON * abs(sp_val)

        var ampl = (0.75 + ca.val) / sqrty

        var result = SFSResult()
        result.val = ampl * sp_val
        result.err = abs(sp_val) * ca.err / sqrty + abs(ampl) * sp_err
        result.err += MSL_DBL_EPSILON * abs(result.val)
        return result^


# ===----------------------------------------------------------------------=== #
# Bessel Y1
# ===----------------------------------------------------------------------=== #


def bessel_Y1(x: Float64) -> SFSResult:
    if x <= 0.0:
        var result = SFSResult()
        result.val = 0.0
        result.err = 0.0
        return result

    var y = x

    if y < 2.0 * MSL_SQRT_DBL_EPSILON:
        var result = SFSResult()
        result.val = -(2.0 / MSL_PI) / y
        result.err = 1.0
        return result
    elif y <= 4.0:
        var by1_c: InlineArray[Float64, 13] = [
            -0.0048476478836790,
            -2.251232040233957,
            0.541141121133259,
            -0.0862003218862530,
            0.0100995839842959,
            -0.000783583778021,
            0.0000423277692699,
            -0.0000017134883750,
            0.0000000538505301,
            -0.0000000013529069,
            0.0000000000279303,
            -0.0000000000004770,
            0.0000000000000068,
        ]
        var z = 0.125 * y * y - 1.0
        var result_c = cheb_eval(by1_c, 12, -1.0, 1.0, z)

        var j1 = bessel_J1(y)

        var result = SFSResult()
        result.val = y * result_c.val + (2.0 / MSL_PI) * (
            j1.val / y - bessel_J0(y).val
        )
        result.err = y * result_c.err + abs(2.0 / MSL_PI) * (
            abs(j1.err / y) + bessel_J0(y).err
        )
        result.err += MSL_DBL_EPSILON * abs(result.val)
        return result^
    else:
        var z = 32.0 / (y * y) - 1.0
        var sqrty = sqrt(y)

        var bm1_c: InlineArray[Float64, 21] = [
            0.1170304509391578,
            -0.0021846037119180,
            0.0000644125585904,
            -0.0000051198594778,
            0.0000005750282432,
            -0.0000000851897998,
            0.0000000154924449,
            -0.0000000032943850,
            0.0000000007901330,
            -0.0000000002089260,
            0.0000000000597373,
            -0.0000000000181477,
            0.0000000000057943,
            -0.0000000000019334,
            0.0000000000006704,
            -0.0000000000002402,
            0.0000000000000887,
            -0.0000000000000336,
            0.0000000000000130,
            -0.0000000000000051,
            0.0000000000000020,
        ]
        var ca = cheb_eval(bm1_c, 20, -1.0, 1.0, z)

        var bth1_c: InlineArray[Float64, 24] = [
            -0.0789742881163768,
            0.0020532235061588,
            -0.0001144014099977,
            0.0000098938054817,
            -0.0000012209770431,
            0.0000001863854547,
            -0.0000000340269543,
            0.0000000071604194,
            -0.0000000016934885,
            0.0000000004406101,
            -0.0000000001236186,
            0.0000000000368366,
            -0.0000000000115636,
            0.0000000000037983,
            -0.0000000000012947,
            0.0000000000004574,
            -0.0000000000001667,
            0.0000000000000625,
            -0.0000000000000240,
            0.0000000000000094,
            -0.0000000000000038,
            0.0000000000000015,
            -0.0000000000000006,
        ]
        var ct = cheb_eval(bth1_c, 23, -1.0, 1.0, z)

        var theta = ct.val / y
        var sp_val = sin(y - 3.0 * MSL_PI_4 + theta)
        var sp_err = MSL_DBL_EPSILON * abs(sp_val)

        var ampl = (0.375 + ca.val) / sqrty

        var result = SFSResult()
        result.val = ampl * sp_val
        result.err = abs(ampl) * sp_err + abs(sp_val) * ca.err / sqrty
        result.err += MSL_DBL_EPSILON * abs(result.val)
        return result^
