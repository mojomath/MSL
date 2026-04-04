# SPDX-License-Identifier: GPL-3.0-or-later

# ===----------------------------------------------------------------------=== #
# MSL (Mojo Scientific Library)
#
# Derived from GNU Scientific Library (GSL)
# Original file: specfunc/airy.c
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
Airy functions Ai(x) and Bi(x) and their derivatives.

This is a direct port of the GSL special functions implementation.
"""

from msl.core.const import MSL_DBL_EPSILON, MSL_PI_4
from msl.sf.result import SFSResult
from std.math import sqrt, exp, cos, sin, abs
from std.collections import InlineArray


comptime SFPrecisionDouble: Int = 0
comptime SFPrecisionSingle: Int = 1
comptime SFPrecisionApprox: Int = 2


# ===----------------------------------------------------------------------=== #
# Chebyshev series evaluation
# ===----------------------------------------------------------------------=== #


def cheb_eval_mode[
    N: Int,
    //,
](
    c: InlineArray[Float64, N],
    order: Int,
    a: Float64,
    b: Float64,
    order_sp: Int,
    x: Float64,
    mode: Int,
) -> SFSResult:
    var d: Float64 = 0.0
    var dd: Float64 = 0.0

    var y = (2.0 * x - a - b) / (b - a)
    var y2 = 2.0 * y

    var eval_order: Int
    if mode == SFPrecisionDouble:
        eval_order = order
    else:
        eval_order = order_sp

    var j = eval_order
    while j >= 1:
        var temp = d
        d = y2 * d - dd + c[j]
        dd = temp
        j -= 1

    var result = SFSResult()
    result.val = y * d - dd + 0.5 * c[0]
    result.err = MSL_DBL_EPSILON * abs(result.val) + abs(c[eval_order])
    return result^


# ===----------------------------------------------------------------------=== #
# Helper functions
# ===----------------------------------------------------------------------=== #


def airy_mod_phase(x: Float64, mode: Int) -> Tuple[SFSResult, SFSResult]:
    # TODO: Figure out why compiler throws error here (same unused error pops up everywhere),
    # This used to work in previous mojo version. Also make sure removing this doesn't cause any issues.
    # var result_m = SFSResult()
    # var result_p = SFSResult()

    var z: Float64
    var a: Float64 = -1.0
    var b: Float64 = 1.0

    if x < -2.0:
        z = 16.0 / (x * x * x) + 1.0

        var am21_c: InlineArray[Float64, 37] = [
            0.0065809191761485,
            0.0023675984685722,
            0.0001324741670371,
            0.0000157600904043,
            0.0000027529702663,
            0.0000006102679017,
            0.0000001595088468,
            0.0000000471033947,
            0.0000000152933871,
            0.0000000053590722,
            0.0000000020000910,
            0.0000000007872292,
            0.0000000003243103,
            0.0000000001390106,
            0.0000000000617011,
            0.0000000000282491,
            0.0000000000132979,
            0.0000000000064188,
            0.0000000000031697,
            0.0000000000015981,
            0.0000000000008213,
            0.0000000000004296,
            0.0000000000002284,
            0.0000000000001232,
            0.0000000000000675,
            0.0000000000000374,
            0.0000000000000210,
            0.0000000000000119,
            0.0000000000000068,
            0.0000000000000039,
            0.0000000000000023,
            0.0000000000000013,
            0.0000000000000008,
            0.0000000000000005,
            0.0000000000000003,
            0.0000000000000001,
            0.0000000000000001,
        ]
        result_m = cheb_eval_mode(am21_c, 36, a, b, 20, z, mode)

        var ath1_c: InlineArray[Float64, 36] = [
            -0.07125837815669365,
            -0.00590471979831451,
            -0.00012114544069499,
            -0.00000988608542270,
            -0.00000138084097352,
            -0.00000026142640172,
            -0.00000006050432589,
            -0.00000001618436223,
            -0.00000000483464911,
            -0.00000000157655272,
            -0.00000000055231518,
            -0.00000000020545441,
            -0.00000000008043412,
            -0.00000000003291252,
            -0.00000000001399875,
            -0.00000000000616151,
            -0.00000000000279614,
            -0.00000000000130428,
            -0.00000000000062373,
            -0.00000000000030512,
            -0.00000000000015239,
            -0.00000000000007758,
            -0.00000000000004020,
            -0.00000000000002117,
            -0.00000000000001132,
            -0.00000000000000614,
            -0.00000000000000337,
            -0.00000000000000188,
            -0.00000000000000105,
            -0.00000000000000060,
            -0.00000000000000034,
            -0.00000000000000020,
            -0.00000000000000011,
            -0.00000000000000007,
            -0.00000000000000004,
            -0.00000000000000002,
        ]
        result_p = cheb_eval_mode(ath1_c, 35, a, b, 15, z, mode)

    elif x <= -1.0:
        z = (16.0 / (x * x * x) + 9.0) / 7.0

        var am22_c: InlineArray[Float64, 33] = [
            -0.01562844480625341,
            0.00778336445239681,
            0.00086705777047718,
            0.00015696627315611,
            0.00003563962571432,
            0.00000924598335425,
            0.00000262110161850,
            0.00000079188221651,
            0.00000025104152792,
            0.00000008265223206,
            0.00000002805711662,
            0.00000000976821090,
            0.00000000347407923,
            0.00000000125828132,
            0.00000000046298826,
            0.00000000017272825,
            0.00000000006523192,
            0.00000000002490471,
            0.00000000000960156,
            0.00000000000373448,
            0.00000000000146417,
            0.00000000000057826,
            0.00000000000022991,
            0.00000000000009197,
            0.00000000000003700,
            0.00000000000001496,
            0.00000000000608,
            0.00000000000000248,
            0.00000000000000101,
            0.00000000000000041,
            0.00000000000000017,
            0.00000000000000007,
            0.00000000000000002,
        ]
        result_m = cheb_eval_mode(am22_c, 32, a, b, 15, z, mode)

        var ath2_c: InlineArray[Float64, 32] = [
            0.00440527345871877,
            -0.03042919452318455,
            -0.00138565328377179,
            -0.00018044439089549,
            -0.00003380847108327,
            -0.00000767818353522,
            -0.00000196783944371,
            -0.00000054837271158,
            -0.00000016254615505,
            -0.00000005053049981,
            -0.00000001631580701,
            -0.00000000543420411,
            -0.00000000185739855,
            -0.00000000064895120,
            -0.00000000023105948,
            -0.00000000008363282,
            -0.00000000003071196,
            -0.00000000001142367,
            -0.00000000000429811,
            -0.00000000000163389,
            -0.00000000000062693,
            -0.00000000000024260,
            -0.00000000000009461,
            -0.00000000000003716,
            -0.00000000000001469,
            -0.00000000000000584,
            -0.00000000000000233,
            -0.00000000000000093,
            -0.00000000000000037,
            -0.00000000000000015,
            -0.00000000000000006,
            -0.00000000000000002,
        ]
        result_p = cheb_eval_mode(ath2_c, 31, a, b, 16, z, mode)
    else:
        return (SFSResult(0.0, 0.0), SFSResult(0.0, 0.0))

    var m = 0.3125 + result_m.val
    var p = -0.625 + result_p.val
    var sqx = sqrt(-x)

    var mod = SFSResult()
    var phase = SFSResult()
    mod.val = sqrt(m / sqx)
    mod.err = abs(mod.val) * (
        MSL_DBL_EPSILON + abs(result_m.err / result_m.val)
    )
    phase.val = MSL_PI_4 - x * sqx * p
    phase.err = abs(phase.val) * (
        MSL_DBL_EPSILON + abs(result_p.err / result_p.val)
    )

    return (mod^, phase^)


def airy_aie(x: Float64, mode: Int) -> SFSResult:
    var sqx = sqrt(x)
    var z = 2.0 / (x * sqx) - 1.0
    var y = sqrt(sqx)

    var aip_c: InlineArray[Float64, 12] = [
        0.0703125,
        -0.23236044689120375,
        -0.17280839208486408,
        -0.04413631820376054,
        -0.00789115823325302,
        -0.00123227172217886,
        -0.00016985419524351,
        -0.00002055842267024,
        -0.00000219365210908,
        -0.00000020561080923,
        -0.00000001684929847,
        -0.00000000120652576,
    ]
    var result_c = cheb_eval_mode(aip_c, 11, -1.0, 1.0, 8, z, mode)

    var result = SFSResult()
    result.val = (0.28125 + result_c.val) / y
    result.err = result_c.err / y + MSL_DBL_EPSILON * abs(result.val)
    return result^


def airy_bie(x: Float64, mode: Int) -> SFSResult:
    var ATR = 8.7506905708484345
    var BTR = -2.0938363213560543

    var result = SFSResult()
    var z: Float64

    if x < 4.0:
        var sqx = sqrt(x)
        z = ATR / (x * sqx) + BTR
        var y = sqrt(sqx)

        var bip_c: InlineArray[Float64, 12] = [
            0.625,
            1.0794554296523708,
            0.4497656761576845,
            0.1063659635665385,
            0.02106664127139533,
            0.00367932139886029,
            0.00056937708959058,
            0.00007783558737798,
            0.00000942155111777,
            0.00000101222930279,
            9.636538e-8,
            8.139e-9,
        ]
        var result_c = cheb_eval_mode(bip_c, 11, -1.0, 1.0, 8, z, mode)
        result.val = (0.625 + result_c.val) / y
        result.err = result_c.err / y + MSL_DBL_EPSILON * abs(result.val)
    else:
        var sqx = sqrt(x)
        z = 16.0 / (x * sqx) - 1.0
        var y = sqrt(sqx)

        var bip2_c: InlineArray[Float64, 13] = [
            0.625,
            1.31307720189604,
            0.42626309709883,
            0.09020063657790,
            0.01609824741832,
            0.00253037715580,
            0.00035145312261,
            0.00004296712290,
            0.00000462320689,
            0.00000043812862,
            3.639e-8,
            2.639e-9,
            1.6e-10,
        ]
        var result_c = cheb_eval_mode(bip2_c, 12, -1.0, 1.0, 9, z, mode)
        result.val = (0.625 + result_c.val) / y
        result.err = result_c.err / y + MSL_DBL_EPSILON * abs(result.val)
    return result^


def sf_cos_err(x: Float64, dx: Float64) -> SFSResult:
    var result = SFSResult()
    result.val = cos(x)
    result.err = abs(dx) * abs(sin(x))
    result.err += MSL_DBL_EPSILON * abs(result.val)
    return result^


def sf_sin_err(x: Float64, dx: Float64) -> SFSResult:
    var result = SFSResult()
    result.val = sin(x)
    result.err = abs(dx) * abs(cos(x))
    result.err += MSL_DBL_EPSILON * abs(result.val)
    return result^


# ===----------------------------------------------------------------------=== #
# Public API
# ===----------------------------------------------------------------------=== #


def airy_ai(x: Float64, mode: Int = SFPrecisionDouble) -> SFSResult:
    if x < -1.0:
        _ = SFSResult()
        _ = SFSResult()
        var (mod, theta) = airy_mod_phase(x, mode)
        var cos_result = sf_cos_err(theta.val, theta.err)

        var result = SFSResult()
        result.val = mod.val * cos_result.val
        result.err = abs(mod.val * cos_result.err) + abs(
            cos_result.val * mod.err
        )
        result.err += MSL_DBL_EPSILON * abs(result.val)
        return result
    elif x <= 1.0:
        var z = x * x * x

        var aif_c: InlineArray[Float64, 9] = [
            -0.03797135849666999750,
            0.05919188853726363857,
            0.00098629280577279975,
            0.00000684884381907656,
            0.00000002594202596219,
            0.00000000006176612774,
            0.00000000000010092454,
            0.00000000000000012014,
            0.00000000000000000010,
        ]
        var result_c0 = cheb_eval_mode(aif_c, 8, -1.0, 1.0, 8, z, mode)

        var aig_c: InlineArray[Float64, 8] = [
            0.01815236558116127,
            0.02157256316601076,
            0.00025678356987483,
            0.00000142652141197,
            0.00000000457211492,
            0.00000000000952517,
            0.00000000000001392,
            0.00000000000000001,
        ]
        var result_c1 = cheb_eval_mode(aig_c, 7, -1.0, 1.0, 7, z, mode)

        var result = SFSResult()
        result.val = 0.375 + (result_c0.val - x * (0.25 + result_c1.val))
        result.err = result_c0.err + abs(x * result_c1.err)
        result.err += MSL_DBL_EPSILON * abs(result.val)
        return result
    else:
        var x32 = x * sqrt(x)
        var s = exp(-2.0 * x32 / 3.0)
        var result_aie = airy_aie(x, mode)

        var result = SFSResult()
        result.val = result_aie.val * s
        result.err = (
            result_aie.err * s + abs(result.val) * x32 * MSL_DBL_EPSILON
        )
        result.err += MSL_DBL_EPSILON * abs(result.val)
        return result


def airy_bi(x: Float64, mode: Int = SFPrecisionDouble) -> SFSResult:
    if x < -1.0:
        _ = SFSResult()
        _ = SFSResult()
        var (mod, theta) = airy_mod_phase(x, mode)
        var sin_result = sf_sin_err(theta.val, theta.err)

        var result = SFSResult()
        result.val = mod.val * sin_result.val
        result.err = abs(mod.val * sin_result.err) + abs(
            sin_result.val * mod.err
        )
        result.err += MSL_DBL_EPSILON * abs(result.val)
        return result
    elif x <= 1.0:
        var z = x * x * x

        var bif_c: InlineArray[Float64, 9] = [
            -0.01673021647198664948,
            0.10252335834249445610,
            0.00170830925073815165,
            0.00001186254546774468,
            0.00000004493290701779,
            0.00000000010698207143,
            0.00000000000017480643,
            0.00000000000000020810,
            0.00000000000000000018,
        ]
        var result_c0 = cheb_eval_mode(bif_c, 8, -1.0, 1.0, 8, z, mode)

        var big_c: InlineArray[Float64, 8] = [
            0.02246622324857452,
            0.03736477545301955,
            0.00044476218957212,
            0.00000247080756363,
            0.00000000791913533,
            0.00000000001649807,
            0.00000000000002411,
            0.00000000000000002,
        ]
        var result_c1 = cheb_eval_mode(big_c, 7, -1.0, 1.0, 7, z, mode)

        var result = SFSResult()
        result.val = 0.375 + (result_c0.val - x * (0.25 + result_c1.val))
        result.err = result_c0.err + abs(x * result_c1.err)
        result.err += MSL_DBL_EPSILON * abs(result.val)
        return result
    elif x < 2.0:
        var z = 2.0 * x * x * x / (x * x) - 1.0

        var aip_c: InlineArray[Float64, 12] = [
            0.0703125,
            -0.23236044689120375,
            -0.17280839208486408,
            -0.04413631820376054,
            -0.00789115823325302,
            -0.00123227172217886,
            -0.00016985419524351,
            -0.00002055842267024,
            -0.00000219365210908,
            -0.00000020561080923,
            -0.00000001684929847,
            -0.00000000120652576,
        ]
        var result_c0 = cheb_eval_mode(aip_c, 11, -1.0, 1.0, 8, z, mode)

        var aig_c: InlineArray[Float64, 8] = [
            0.01815236558116127,
            0.02157256316601076,
            0.00025678356987483,
            0.00000142652141197,
            0.00000000457211492,
            0.00000000000952517,
            0.00000000000001392,
            0.00000000000000001,
        ]
        var result_c1 = cheb_eval_mode(aig_c, 7, -1.0, 1.0, 7, z, mode)

        var x32 = x * sqrt(x)

        var result = SFSResult()
        result.val = (
            0.625 + result_c0.val - x * (0.375 + result_c1.val)
        ) / sqrt(x32)
        result.err = (result_c0.err + abs(x * result_c1.err)) / sqrt(x32)
        result.err += MSL_DBL_EPSILON * abs(result.val)
        return result
    else:
        var z = 16.0 / (x * sqrt(x)) - 1.0

        var bip_c: InlineArray[Float64, 12] = [
            0.625,
            1.0794554296523708,
            0.4497656761576845,
            0.1063659635665385,
            0.02106664127139533,
            0.00367932139886029,
            0.00056937708959058,
            0.00007783558737798,
            0.00000942155111777,
            0.00000101222930279,
            9.636538e-8,
            8.139e-9,
        ]
        var result_c0 = cheb_eval_mode(bip_c, 11, -1.0, 1.0, 8, z, mode)

        var big_c: InlineArray[Float64, 8] = [
            0.02246622324857452,
            0.03736477545301955,
            0.00044476218957212,
            0.00000247080756363,
            0.00000000791913533,
            0.00000000001649807,
            0.00000000000002411,
            0.00000000000000002,
        ]
        var result_c1 = cheb_eval_mode(big_c, 7, -1.0, 1.0, 7, z, mode)

        var y = sqrt(sqrt(x))

        var result = SFSResult()
        result.val = (1.875 + result_c0.val - x * (0.625 + result_c1.val)) / y
        result.err = (result_c0.err + abs(x * result_c1.err)) / y
        result.err += MSL_DBL_EPSILON * abs(result.val)
        return result


def airy_ai_scaled(x: Float64, mode: Int = SFPrecisionDouble) -> SFSResult:
    if x < -1.0:
        _ = SFSResult()
        _ = SFSResult()
        var (mod, theta) = airy_mod_phase(x, mode)
        var cos_result = sf_cos_err(theta.val, theta.err)

        var result = SFSResult()
        result.val = mod.val * cos_result.val
        result.err = abs(mod.val * cos_result.err) + abs(
            cos_result.val * mod.err
        )
        result.err += MSL_DBL_EPSILON * abs(result.val)
        return result
    elif x <= 1.0:
        var z = x * x * x

        var aif_c: InlineArray[Float64, 9] = [
            -0.03797135849666999750,
            0.05919188853726363857,
            0.00098629280577279975,
            0.00000684884381907656,
            0.00000002594202596219,
            0.00000000006176612774,
            0.00000000000010092454,
            0.00000000000000012014,
            0.00000000000000000010,
        ]
        var result_c0 = cheb_eval_mode(aif_c, 8, -1.0, 1.0, 8, z, mode)

        var aig_c: InlineArray[Float64, 8] = [
            0.01815236558116127,
            0.02157256316601076,
            0.00025678356987483,
            0.00000142652141197,
            0.00000000457211492,
            0.00000000000952517,
            0.00000000000001392,
            0.00000000000000001,
        ]
        var result_c1 = cheb_eval_mode(aig_c, 7, -1.0, 1.0, 7, z, mode)

        var result = SFSResult()
        result.val = 0.375 + (result_c0.val - x * (0.25 + result_c1.val))
        result.err = result_c0.err + abs(x * result_c1.err)
        result.err += MSL_DBL_EPSILON * abs(result.val)

        if x > 0.0:
            var scale = exp(2.0 / 3.0 * sqrt(z))
            result.val = result.val * scale
            result.err = result.err * scale

        return result
    else:
        return airy_aie(x, mode)


def airy_bi_scaled(x: Float64, mode: Int = SFPrecisionDouble) -> SFSResult:
    if x < -1.0:
        _ = SFSResult()
        _ = SFSResult()
        var (mod, theta) = airy_mod_phase(x, mode)
        var sin_result = sf_sin_err(theta.val, theta.err)

        var result = SFSResult()
        result.val = mod.val * sin_result.val
        result.err = abs(mod.val * sin_result.err) + abs(
            sin_result.val * mod.err
        )
        result.err += MSL_DBL_EPSILON * abs(result.val)
        return result
    elif x <= 1.0:
        var z = x * x * x

        var bif_c: InlineArray[Float64, 9] = [
            -0.01673021647198664948,
            0.10252335834249445610,
            0.00170830925073815165,
            0.00001186254546774468,
            0.00000004493290701779,
            0.00000000010698207143,
            0.00000000000017480643,
            0.00000000000000020810,
            0.00000000000000000018,
        ]
        var result_c0 = cheb_eval_mode(bif_c, 8, -1.0, 1.0, 8, z, mode)

        var big_c: InlineArray[Float64, 8] = [
            0.02246622324857452,
            0.03736477545301955,
            0.00044476218957212,
            0.00000247080756363,
            0.00000000791913533,
            0.00000000001649807,
            0.00000000000002411,
            0.00000000000000002,
        ]
        var result_c1 = cheb_eval_mode(big_c, 7, -1.0, 1.0, 7, z, mode)

        var result = SFSResult()
        result.val = 0.375 + (result_c0.val - x * (0.25 + result_c1.val))
        result.err = result_c0.err + abs(x * result_c1.err)
        result.err += MSL_DBL_EPSILON * abs(result.val)

        if x > 0.0:
            var scale = exp(-2.0 / 3.0 * sqrt(z))
            result.val = result.val * scale
            result.err = result.err * scale

        return result
    else:
        return airy_bie(x, mode)


def airy_ai_deriv(x: Float64, mode: Int = SFPrecisionDouble) -> SFSResult:
    return airy_ai_deriv_scaled(x, mode)


def airy_bi_deriv(x: Float64, mode: Int = SFPrecisionDouble) -> SFSResult:
    return airy_bi_deriv_scaled(x, mode)


def airy_ai_deriv_scaled(
    x: Float64, mode: Int = SFPrecisionDouble
) -> SFSResult:
    if x < -1.0:
        _ = SFSResult()
        _ = SFSResult()
        var (mod, theta) = airy_mod_phase(x, mode)
        var sin_result = sf_sin_err(theta.val, theta.err)

        var result = SFSResult()
        result.val = -mod.val * sin_result.val
        result.err = abs(mod.val * sin_result.err) + abs(
            sin_result.val * mod.err
        )
        result.err += MSL_DBL_EPSILON * abs(result.val)
        return result
    elif x <= 1.0:
        var result = SFSResult()

        if abs(x) < MSL_DBL_EPSILON:
            result.val = -0.2588194037724879
            result.err = MSL_DBL_EPSILON
            return result

        var x2 = x * x
        var z = x2 * x2

        var aif_c: InlineArray[Float64, 9] = [
            -0.03797135849666999750,
            0.05919188853726363857,
            0.00098629280577279975,
            0.00000684884381907656,
            0.00000002594202596219,
            0.00000000006176612774,
            0.00000000000010092454,
            0.00000000000000012014,
            0.00000000000000000010,
        ]
        var c0 = cheb_eval_mode(aif_c, 8, -1.0, 1.0, 8, z, mode)

        var aig_c: InlineArray[Float64, 8] = [
            0.01815236558116127,
            0.02157256316601076,
            0.00025678356987483,
            0.00000142652141197,
            0.00000000457211492,
            0.00000000000952517,
            0.00000000000001392,
            0.00000000000000001,
        ]
        var c1 = cheb_eval_mode(aig_c, 7, -1.0, 1.0, 7, z, mode)

        result.val = -x * (0.375 + c0.val + 0.5 * c1.val)
        result.err = (
            c0.err + abs(x * c1.err) + MSL_DBL_EPSILON * abs(result.val)
        )
        return result
    else:
        var sqx = sqrt(x)
        var x32 = x * sqx
        var z = 2.0 / (x * sqx) - 1.0

        var aip1_c: InlineArray[Float64, 9] = [
            -0.375,
            0.3745990447334779,
            0.1928507877655783,
            0.0475356697641584,
            0.0091569801738944,
            0.0015463410408915,
            0.0002313097224174,
            3.053e-5,
            3.58e-6,
        ]
        var c0 = cheb_eval_mode(aip1_c, 8, -1.0, 1.0, 6, z, mode)

        var result = SFSResult()
        var s = exp(-2.0 * x32 / 3.0)
        result.val = (-0.140625 + c0.val) * s / sqx
        result.err = c0.err * s / sqx + abs(result.val) * x32 * MSL_DBL_EPSILON
        result.err += MSL_DBL_EPSILON * abs(result.val)
        return result


def airy_bi_deriv_scaled(
    x: Float64, mode: Int = SFPrecisionDouble
) -> SFSResult:
    if x < -1.0:
        _ = SFSResult()
        _ = SFSResult()
        var (mod, theta) = airy_mod_phase(x, mode)
        var cos_result = sf_cos_err(theta.val, theta.err)

        var result = SFSResult()
        result.val = mod.val * cos_result.val
        result.err = abs(mod.val * cos_result.err) + abs(
            cos_result.val * mod.err
        )
        result.err += MSL_DBL_EPSILON * abs(result.val)
        return result
    elif x <= 1.0:
        var result = SFSResult()

        if abs(x) < MSL_DBL_EPSILON:
            result.val = 0.4482883573538263
            result.err = MSL_DBL_EPSILON
            return result

        var x2 = x * x
        var z = x2 * x2

        var bif_c: InlineArray[Float64, 9] = [
            -0.01673021647198664948,
            0.10252335834249445610,
            0.00170830925073815165,
            0.00001186254546774468,
            0.00000004493290701779,
            0.00000000010698207143,
            0.00000000000017480643,
            0.00000000000000020810,
            0.00000000000000000018,
        ]
        var c0 = cheb_eval_mode(bif_c, 8, -1.0, 1.0, 8, z, mode)

        var big_c: InlineArray[Float64, 8] = [
            0.02246622324857452,
            0.03736477545301955,
            0.00044476218957212,
            0.00000247080756363,
            0.00000000791913533,
            0.00000000001649807,
            0.00000000000002411,
            0.00000000000000002,
        ]
        var c1 = cheb_eval_mode(big_c, 7, -1.0, 1.0, 7, z, mode)

        result.val = x * (0.375 + c0.val - 0.5 * c1.val)
        result.err = (
            c0.err + abs(x * c1.err) + MSL_DBL_EPSILON * abs(result.val)
        )
        return result
    else:
        var result = SFSResult()

        if x < 2.0:
            var sqx = sqrt(x)
            var z = 8.7506905708484345 / (x * sqx) - 2.0938363213560543

            var bip1_c: InlineArray[Float64, 9] = [
                0.375,
                0.2713721561297926,
                0.1250593785093774,
                0.0392457382621135,
                0.0097658452335542,
                0.0020969763795389,
                0.0003945661262184,
                6.556e-5,
                9.63e-6,
            ]
            var c0 = cheb_eval_mode(bip1_c, 8, -1.0, 1.0, 6, z, mode)
            result.val = (0.140625 + c0.val) / (sqx * sqx * sqx)
            result.err = c0.err / (sqx * sqx * sqx) + MSL_DBL_EPSILON * abs(
                result.val
            )
        else:
            var sqx = sqrt(x)
            var z = 16.0 / (x * sqx) - 1.0

            var bip1_c: InlineArray[Float64, 9] = [
                0.375,
                0.2713721561297926,
                0.1250593785093774,
                0.0392457382621135,
                0.0097658452335542,
                0.0020969763795389,
                0.0003945661262184,
                6.556e-5,
                9.63e-6,
            ]
            var c0 = cheb_eval_mode(bip1_c, 8, -1.0, 1.0, 6, z, mode)
            result.val = (0.140625 + c0.val) / (sqx * sqx * sqx)
            result.err = c0.err / (sqx * sqx * sqx) + MSL_DBL_EPSILON * abs(
                result.val
            )

        return result
