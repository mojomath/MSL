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
from msl.core.errno import MSL_EDOM
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
        result.errno = MSL_EDOM
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
        result.errno = MSL_EDOM
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


# ===----------------------------------------------------------------------=== #
# Modified Bessel I0
# ===----------------------------------------------------------------------=== #


def bessel_I0(x: Float64) -> SFSResult:
    var y = abs(x)

    if y < 2.0 * MSL_SQRT_DBL_EPSILON:
        var result = SFSResult()
        result.val = 1.0
        result.err = y * y
        return result
    elif y <= 4.0:
        var bi0_c: InlineArray[Float64, 13] = [
            0.0441891604541899,
            0.450358340603743,
            0.127474324165233,
            0.0223152542422695,
            0.0026566796608573,
            0.0002305194814519,
            0.0000138599875892,
            0.0000006022330420,
            0.0000000193443772,
            0.0000000004728274,
            0.0000000000089555,
            0.0000000000001334,
            0.0000000000000016,
        ]
        var z = 0.125 * y * y - 1.0
        return cheb_eval(bi0_c, 12, -1.0, 1.0, z)
    else:
        var z = 16.0 / (y * y) - 1.0

        var bim0_c: InlineArray[Float64, 19] = [
            0.249505230526282,
            -0.0000412666474633,
            0.0000018687834231,
            -0.0000001765437648,
            0.0000000245654224,
            -0.0000000043395356,
            0.0000000008913655,
            -0.0000000002045838,
            0.0000000000514187,
            -0.0000000000137784,
            0.0000000000038608,
            -0.0000000000011190,
            0.0000000000003330,
            -0.0000000000001013,
            0.0000000000000314,
            -0.0000000000000099,
            0.0000000000000031,
            -0.0000000000000010,
            0.0000000000000003,
        ]
        var ca = cheb_eval(bim0_c, 18, -1.0, 1.0, z)

        var sqrty = sqrt(y)
        var result = SFSResult()
        result.val = (0.5 + ca.val) / sqrty * exp(y)
        result.err = (
            (ca.err / sqrty + (0.5 + ca.val) / sqrty) * MSL_DBL_EPSILON * exp(y)
        )
        result.err += MSL_DBL_EPSILON * abs(result.val)
        return result^


# ===----------------------------------------------------------------------=== #
# Modified Bessel I1
# ===----------------------------------------------------------------------=== #


def bessel_I1(x: Float64) -> SFSResult:
    var y = abs(x)
    var sign_x = 1.0 if x > 0.0 else -1.0

    if y < 2.0 * MSL_SQRT_DBL_EPSILON:
        var result = SFSResult()
        result.val = 0.5 * x
        result.err = y * y
        return result
    elif y <= 4.0:
        var bi1_c: InlineArray[Float64, 13] = [
            0.0072995592376945,
            0.260805308193142,
            0.115318456458377,
            0.0200950406701093,
            0.0023849779372592,
            0.0002044278900296,
            0.0000121892097220,
            0.0000005275476211,
            0.0000000169309773,
            0.0000000004112517,
            0.0000000000077766,
            0.0000000000001157,
            0.0000000000000014,
        ]
        var z = 0.125 * y * y - 1.0
        var result_c = cheb_eval(bi1_c, 12, -1.0, 1.0, z)

        var result = SFSResult()
        result.val = sign_x * y * result_c.val
        result.err = y * result_c.err
        result.err += MSL_DBL_EPSILON * abs(result.val)
        return result^
    else:
        var z = 16.0 / (y * y) - 1.0

        var bim1_c: InlineArray[Float64, 19] = [
            0.333095859022228,
            -0.001469084870662,
            0.0000933771327677,
            -0.0000098974980950,
            0.0000014571696562,
            -0.0000002693803777,
            0.0000000563779537,
            -0.0000000129905813,
            0.0000000031635988,
            -0.0000000008022816,
            0.0000000002095084,
            -0.0000000000559992,
            0.0000000000152400,
            -0.0000000000042255,
            0.0000000000011886,
            -0.0000000000003390,
            0.0000000000000976,
            -0.0000000000000283,
            0.0000000000000083,
        ]
        var ca = cheb_eval(bim1_c, 18, -1.0, 1.0, z)

        var sqrty = sqrt(y)
        var result = SFSResult()
        result.val = sign_x * (0.25 + ca.val) / sqrty * exp(y)
        result.err = (
            ((ca.err + abs(ca.err)) / sqrty + abs(0.25 + ca.val) / sqrty)
            * MSL_DBL_EPSILON
            * exp(y)
        )
        result.err += MSL_DBL_EPSILON * abs(result.val)
        return result^


# ===----------------------------------------------------------------------=== #
# Modified Bessel K0
# ===----------------------------------------------------------------------=== #


def bessel_K0(x: Float64) -> SFSResult:
    if x <= 0.0:
        var result = SFSResult()
        result.errno = MSL_EDOM
        return result

    var y = x

    if y < 2.0 * MSL_SQRT_DBL_EPSILON:
        var result = SFSResult()
        result.val = -log(y)
        result.err = 1.0
        return result
    elif y <= 4.0:
        var bk0_c: InlineArray[Float64, 13] = [
            0.0129745833230940,
            -0.104404983299170,
            0.0356689352347854,
            -0.0061599960445826,
            0.0006109842484084,
            -0.0000420143327697,
            0.0000020996202954,
            -0.0000000789347220,
            0.0000000022823612,
            -0.0000000000514181,
            0.0000000000009142,
            -0.0000000000000129,
            0.0000000000000001,
        ]
        var z = 0.125 * y * y - 1.0
        var result_c = cheb_eval(bk0_c, 12, -1.0, 1.0, z)

        var result = SFSResult()
        result.val = result_c.val - log(y) * bessel_I0(y).val
        result.err = result_c.err + abs(log(y)) * bessel_I0(y).err
        result.err += MSL_DBL_EPSILON * abs(result.val)
        return result^
    else:
        var z = 16.0 / (y * y) - 1.0

        var bkm0_c: InlineArray[Float64, 19] = [
            0.159182408570797,
            -0.0022339519477607,
            0.0000987613446424,
            -0.0000105838269824,
            0.0000015872216350,
            -0.0000002956231099,
            0.0000000621777414,
            -0.0000000142523739,
            0.0000000034615829,
            -0.0000000008759293,
            0.0000000002278259,
            -0.0000000000604932,
            0.0000000000163763,
            -0.0000000000045098,
            0.0000000000012617,
            -0.0000000000003584,
            0.0000000000001030,
            -0.0000000000000299,
            0.0000000000000088,
        ]
        var ca = cheb_eval(bkm0_c, 18, -1.0, 1.0, z)

        var sqrty = sqrt(y)
        var result = SFSResult()
        result.val = (0.5 + ca.val) / sqrty * exp(-y)
        result.err = (
            (ca.err / sqrty + (0.5 + ca.val) / sqrty)
            * MSL_DBL_EPSILON
            * exp(-y)
        )
        result.err += MSL_DBL_EPSILON * abs(result.val)
        return result^


# ===----------------------------------------------------------------------=== #
# Modified Bessel K1
# ===----------------------------------------------------------------------=== #


def bessel_K1(x: Float64) -> SFSResult:
    if x <= 0.0:
        var result = SFSResult()
        result.errno = MSL_EDOM
        return result

    var y = x

    if y < 2.0 * MSL_SQRT_DBL_EPSILON:
        var result = SFSResult()
        result.val = 1.0 / y
        result.err = 1.0
        return result
    elif y <= 4.0:
        var bk1_c: InlineArray[Float64, 13] = [
            -0.0014377633780934,
            -0.150226342304776,
            0.0865054235600797,
            -0.0190266048042478,
            0.0025332448505834,
            -0.0002343777456214,
            0.0000155006225212,
            -0.0000007673684973,
            0.0000000295213789,
            -0.0000000008902698,
            0.0000000000212146,
            -0.0000000000004000,
            0.0000000000000060,
        ]
        var z = 0.125 * y * y - 1.0
        var result_c = cheb_eval(bk1_c, 12, -1.0, 1.0, z)

        var i1 = bessel_I1(y)

        var result = SFSResult()
        result.val = y * result_c.val + (i1.val / y + bessel_I0(y).val)
        result.err = y * result_c.err + abs(i1.err / y + bessel_I0(y).err)
        result.err += MSL_DBL_EPSILON * abs(result.val)
        return result^
    else:
        var z = 16.0 / (y * y) - 1.0

        var bkm1_c: InlineArray[Float64, 19] = [
            0.218850767684584,
            -0.0055468183653642,
            0.0003303857646899,
            -0.0000430440957271,
            0.0000072492693760,
            -0.0000014931612184,
            0.0000003437421553,
            -0.0000000855226529,
            0.0000000223552520,
            -0.0000000060605860,
            0.0000000016768140,
            -0.0000000004719685,
            0.0000000001346794,
            -0.0000000000389725,
            0.0000000000114188,
            -0.0000000000033783,
            0.0000000000010079,
            -0.0000000000003030,
            0.0000000000000917,
        ]
        var ca = cheb_eval(bkm1_c, 18, -1.0, 1.0, z)

        var sqrty = sqrt(y)
        var result = SFSResult()
        result.val = (0.75 + ca.val) / sqrty * exp(-y)
        result.err = (
            (ca.err / sqrty + (0.75 + ca.val) / sqrty)
            * MSL_DBL_EPSILON
            * exp(-y)
        )
        result.err += MSL_DBL_EPSILON * abs(result.val)
        return result^


# ===----------------------------------------------------------------------=== #
# Bessel Jn (integer order)
# ===----------------------------------------------------------------------=== #


def bessel_Jn(n: Int, x: Float64) -> SFSResult:
    var sign: Float64 = 1.0
    var nn = n

    if nn < 0:
        nn = -nn
        if nn % 2 == 1:
            sign = -1.0

    var xx = x
    if xx < 0.0:
        xx = -x
        if nn % 2 == 1:
            sign = -1.0

    if nn == 0:
        return bessel_J0(xx)
    elif nn == 1:
        return bessel_J1(xx)
    else:
        if xx == 0.0:
            var result = SFSResult()
            result.val = 0.0
            result.err = 0.0
            return result^

        var nfp1 = Float64(nn + 1)
        if xx * xx < 10.0 * nfp1 * MSL_DBL_EPSILON * 5.0:
            var result = SFSResult()
            result.val = 0.0
            result.err = MSL_DBL_EPSILON
            return result^

        var ratio: Float64
        var a: Float64 = 2.0 * Float64(nn) / xx
        var b: Float64 = 2.0 / xx
        var c: Float64
        var d: Float64 = 1.0
        var i = 0
        while i < 300:
            c = b + a * d
            d = 1.0 / c
            var new_d = b * d
            if abs(new_d - 1.0) < 1e-30:
                break
            a = a + b * b
            i += 1

        ratio = d
        # sgn = c

        var Jkp1: Float64 = 1e-190 * ratio
        var Jk: Float64 = 1e-190
        var Jkm1: Float64

        var k = nn
        while k > 0:
            Jkm1 = 2.0 * Float64(k) / xx * Jk - Jkp1
            Jkp1 = Jk
            Jk = Jkm1
            k -= 1

        var ans: Float64
        var err: Float64

        if abs(Jkp1) > abs(Jk):
            var b1 = bessel_J1(xx)
            ans = b1.val / Jkp1 * 1e-190
            err = b1.err / Jkp1 * 1e-190
        else:
            var b0 = bessel_J0(xx)
            ans = b0.val / Jk * 1e-190
            err = b0.err / Jk * 1e-190

        var result = SFSResult()
        result.val = sign * ans
        result.err = abs(err) + MSL_DBL_EPSILON * abs(result.val)
        return result^


def bessel_Jnu_asympx(nu: Float64, x: Float64) -> SFSResult:
    var mu = 4.0 * nu * nu
    var chi = x - (0.5 * nu + 0.25) * MSL_PI

    var P: Float64 = 0.0
    var Q: Float64 = 0.0

    var k: Float64 = 0.0
    var t: Float64 = 1.0
    var convP: Bool
    var convQ: Bool

    while k < 1000.0:
        if k == 0.0:
            t = 1.0
        else:
            t = -(mu - (2.0 * k - 1.0) * (2.0 * k - 1.0)) / (k * 8.0 * x)

        convP = abs(t) < MSL_DBL_EPSILON * abs(P)
        P += t

        k += 1.0

        t *= (mu - (2.0 * k - 1.0) * (2.0 * k - 1.0)) / (k * 8.0 * x)
        convQ = abs(t) < MSL_DBL_EPSILON * abs(Q)
        Q += t

        if convP and convQ and k > nu / 2.0:
            break

        k += 1.0

    var pre = sqrt(2.0 / (MSL_PI * x))
    var c = cos(chi)
    var s = sin(chi)

    var result = SFSResult()
    result.val = pre * (c * P - s * Q)
    result.err = (
        pre
        * MSL_DBL_EPSILON
        * (abs(c * P) + abs(s * Q) + abs(t))
        * (1.0 + abs(x))
    )
    return result^


# ===----------------------------------------------------------------------=== #
# Bessel Y_n - upward recurrence Y_{n+1} = (2n/x)*Y_n - Y_{n-1}
# ===----------------------------------------------------------------------=== #
# TODO: Consider making `n` a compile time parameter.


def bessel_Yn(n: Int, x: Float64) -> SFSResult:
    """Bessel function of the second kind, integer order n."""
    var result = SFSResult()
    if x <= 0.0:
        result.errno = MSL_EDOM
        return result^

    if n < 0:
        # Y_{-n}(x) = (-1)^n * Y_n(x)
        var r = bessel_Yn(-n, x)
        if (-n) % 2 == 1:
            r.val = -r.val
        return r^

    if n == 0:
        return bessel_Y0(x)
    if n == 1:
        return bessel_Y1(x)

    var Ykm1 = bessel_Y0(x)
    var Yk = bessel_Y1(x)
    var Ykp1 = SFSResult()

    for k in range(1, n):
        Ykp1.val = (2.0 * Float64(k) / x) * Yk.val - Ykm1.val
        Ykp1.err = (
            (2.0 * Float64(k) / x) * Yk.err
            + Ykm1.err
            + MSL_DBL_EPSILON * abs(Ykp1.val)
        )
        Ykm1 = Yk
        Yk = Ykp1

    result.val = Yk.val
    result.err = Yk.err + Float64(n) * MSL_DBL_EPSILON * abs(Yk.val)
    return result^


# ===----------------------------------------------------------------------=== #
# Modified Bessel I_n - backward recurrence (Miller's algorithm)
# I_{n-1} = I_{n+1} + (2n/x)*I_n
# ===----------------------------------------------------------------------=== #


def bessel_In(n: Int, x: Float64) -> SFSResult:
    """Modified Bessel function of the first kind, integer order n."""
    var result = SFSResult()
    var nn = abs(n) if n < 0 else n
    var sign: Float64 = -1.0 if (n < 0 and nn % 2 == 1) else 1.0

    if x == 0.0:
        result.val = 1.0 if nn == 0 else 0.0
        return result^

    if nn == 0:
        var r = bessel_I0(x)
        r.val *= sign
        return r^
    if nn == 1:
        var r = bessel_I1(x)
        r.val *= sign
        return r^

    # Miller's backward recurrence from n_start down to 0/1
    var n_start = nn + 30  # start well above target
    var Ikp1: Float64 = 0.0
    var Ik: Float64 = 1.0e-190
    var target_val: Float64 = 0.0
    var I0_norm: Float64
    var I1_norm: Float64 = 0.0

    var k = n_start
    while k >= 1:
        var Ikm1 = Ikp1 + (2.0 * Float64(k) / x) * Ik
        Ikp1 = Ik
        Ik = Ikm1
        if k == nn:
            target_val = Ikp1  # this is I_n before normalization
        if k == 1:
            I1_norm = Ikp1
        k -= 1
    I0_norm = Ik  # k=0 value

    # Normalize using exact I0 or I1
    var b0 = bessel_I0(x)
    var b1 = bessel_I1(x)
    var norm: Float64
    if abs(I0_norm) > abs(I1_norm):
        norm = b0.val / I0_norm
    else:
        norm = b1.val / I1_norm

    result.val = sign * target_val * norm
    result.err = abs(norm) * 1.0e-190 + MSL_DBL_EPSILON * abs(
        result.val
    ) * Float64(nn)
    return result^


# ===----------------------------------------------------------------------=== #
# Modified Bessel K_n - upward recurrence K_{n+1} = K_{n-1} + (2n/x)*K_n
# ===----------------------------------------------------------------------=== #


def bessel_Kn(n: Int, x: Float64) -> SFSResult:
    """Modified Bessel function of the second kind, integer order n."""
    var result = SFSResult()
    var nn = abs(n)  # K_n = K_{-n}

    if x <= 0.0:
        result.errno = MSL_EDOM
        return result^

    if nn == 0:
        return bessel_K0(x)
    if nn == 1:
        return bessel_K1(x)

    var Kkm1 = bessel_K0(x)
    var Kk = bessel_K1(x)
    var Kkp1 = SFSResult()

    for k in range(1, nn):
        Kkp1.val = Kkm1.val + (2.0 * Float64(k) / x) * Kk.val
        Kkp1.err = (
            Kkm1.err
            + (2.0 * Float64(k) / x) * Kk.err
            + MSL_DBL_EPSILON * abs(Kkp1.val)
        )
        Kkm1 = Kk
        Kk = Kkp1

    result.val = Kk.val
    result.err = Kk.err + Float64(nn) * MSL_DBL_EPSILON * abs(Kk.val)
    return result^


# ===----------------------------------------------------------------------=== #
# Public API
# ===----------------------------------------------------------------------=== #


def bessel_j0(x: Float64) -> SFSResult:
    if x == 0.0:
        var result = SFSResult()
        result.val = 1.0
        result.err = 0.0
        return result
    return bessel_J0(x)


def bessel_j1(x: Float64) -> SFSResult:
    if x == 0.0:
        var result = SFSResult()
        result.val = 0.0
        result.err = 0.0
        return result
    return bessel_J1(x)


def bessel_y0(x: Float64) -> SFSResult:
    return bessel_Y0(x)


def bessel_y1(x: Float64) -> SFSResult:
    return bessel_Y1(x)


def bessel_i0_scaled(x: Float64) -> SFSResult:
    return bessel_I0(x)


def bessel_i1_scaled(x: Float64) -> SFSResult:
    return bessel_I1(x)


def bessel_k0_scaled(x: Float64) -> SFSResult:
    return bessel_K0(x)


def bessel_k1_scaled(x: Float64) -> SFSResult:
    return bessel_K1(x)
