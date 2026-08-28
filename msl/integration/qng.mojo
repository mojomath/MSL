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

Attempts integration using progressively higher-order rules until tolerance
is met:
  1. 10-point Gauss / 21-point Kronrod  (21 evaluations)
  2. 43-point Kronrod extension          (+22 evaluations, 43 total)
  3. 87-point Kronrod extension          (+44 evaluations, 87 total)

Function evaluations are reused across levels.
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
from msl.integration.workspace import IntegrationResult


# ===----------------------------------------------------------------------=== #
# Quadrature nodes and weights (from QUADPACK qng.f)
# Only positive nodes stored; use f(c+x) + f(c-x) for each.
# ===----------------------------------------------------------------------=== #

# Level 1: nodes shared by 10-, 21-, 43-, 87-point rules
comptime x1: Array[Float64, 5] = [
    0.973906528517171720077964012084452,
    0.865063366688984510732096688423493,
    0.679409568299024406234327365114874,
    0.433395394129247190799265943165784,
    0.148874338981631210884826001129720,
]
comptime w10: Array[Float64, 5] = [
    0.066671344308688137593568809893332,
    0.149451349150580593145776339657697,
    0.219086362515982043995534934228163,
    0.269266719309996355091226921569469,
    0.295524224714752870173892994651338,
]
comptime w21a: Array[Float64, 5] = [
    0.032558162307964727478818972459390,
    0.075039674810919952767043140916190,
    0.109387158802297641899210590325805,
    0.134709217311473325928054001771707,
    0.147739104901338491374841515972068,
]

# Level 2: additional nodes for 21-point rule (+ centre weight)
comptime x2: Array[Float64, 5] = [
    0.995657163025808080735527280689003,
    0.930157491355708226001207180059508,
    0.780817726586416897063717578345042,
    0.562757134668604683339000099272694,
    0.294392862701460198131126603103866,
]
comptime w21b: Array[Float64, 6] = [
    0.011694638867371874278064396062192,
    0.054755896574351996031381300244580,
    0.093125454583697605535065465083366,
    0.123491976262065851077958109831074,
    0.142775938577060080797094273138717,
    0.149445554002916905664936468389821,  # centre
]

# Level 3: additional nodes for 43-point rule (+ centre weight)
comptime x3: Array[Float64, 11] = [
    0.999333360901932097599310220204904,
    0.987433402908088869795961478381209,
    0.954807934814266299257919200290931,
    0.900148695748328293625099494069092,
    0.825198314983114150847066732588521,
    0.732148388989304982612354848755461,
    0.622847970537725238987089396762454,
    0.499479574071056499952214885499755,
    0.364901661346580768767651787430569,
    0.222254919776601296498260928066212,
    0.074650617461383322043914435796506,
]
comptime w43a: Array[Float64, 10] = [
    0.016296734289666564924281974617663,
    0.037522876120869501461613795898115,
    0.054694902058255442147212685465005,
    0.067355414609478086075553166302174,
    0.073870199632393953432140695251367,
    0.005768556059769796184184327908655,
    0.027371890593248842081276069289151,
    0.046560826910428830743339154433824,
    0.061744995201442546108064816314038,
    0.071387267268693397076946364703380,
]
comptime w43b: Array[Float64, 12] = [
    0.001844477640212414100389106552965,
    0.010798689585891651740465406741293,
    0.021895363867795428102523123075149,
    0.032597463975345689443882222526137,
    0.042163137935191738062988305741048,
    0.050741939600184577780189020092085,
    0.058379395542619248375475369330206,
    0.064746404951445885544689259517511,
    0.069566197912356484528633315038405,
    0.072824441471833208430592823983711,
    0.074507751014175118273571813842889,
    0.074722147517403021379539427208517,  # centre
]

# Level 4: additional nodes for 87-point rule (+ centre weight)
comptime x4: Array[Float64, 22] = [
    0.999902977262729234490529830591582,
    0.997989895986678745427496322365960,
    0.992175497860687222808523352251425,
    0.981358163572712422129460800769332,
    0.965057623858384619128284110607926,
    0.943167613133670596816416948429213,
    0.915806414685507070583565900891749,
    0.883221657771316499751925866119908,
    0.845710748462415666605902011504855,
    0.803557658035231265101066990169389,
    0.757005730685495558328942793432020,
    0.706273209787321819828053514318680,
    0.651589466501177922534422205016736,
    0.593223374057961099952690314908337,
    0.531493605970831932285268948562671,
    0.466763623042022809938418619526477,
    0.399424847859218884233258800975867,
    0.329874877106183814931675945537667,
    0.258503559202161169670224915734406,
    0.185695396568346652015917141167606,
    0.111842213179907478300294509297585,
    0.037352123394619963421895427476047,
]
comptime w87a: Array[Float64, 21] = [
    0.008148377384149172900002878448190,
    0.018761438201562822243935059003794,
    0.027347451050052286161582829741283,
    0.033677707311637930046581056957588,
    0.036935099820427907614589586742499,
    0.002884872430211530501334156248695,
    0.013685946022712701888950035273128,
    0.023280413502888311123409291030404,
    0.030872497611713358675466394126442,
    0.035693633639418770719351355457044,
    0.000915283345202241360843392549948,
    0.005399280219300471367738743391053,
    0.010947679601118931134327826856808,
    0.016298731696787335262665703223280,
    0.021081568889203835112433060188190,
    0.025370969769253827243467999831710,
    0.029189697756475752501446154084920,
    0.032373202467202789685788194889595,
    0.034783098950365142750781997949596,
    0.036412220731351787562801163687577,
    0.037253875503047708539592001191226,
]
comptime w87b: Array[Float64, 23] = [
    0.000274145563762072350016527092881,
    0.001807124155057942948341311753254,
    0.004096869282759164864458070683480,
    0.006758290051847378699816577897424,
    0.009549957672201646536053581325377,
    0.012329447652244853694626639963780,
    0.015010447346388952376697286041943,
    0.017548967986243191099665352925900,
    0.019938037786440888202278192730714,
    0.022194935961012286796332102959499,
    0.024339147126000805470360647041454,
    0.026374505414839207241503786552615,
    0.028286910788771200659968002987960,
    0.030052581128092695322521110347341,
    0.031646751371439929404586051078883,
    0.033050413419978503290785944862689,
    0.034255099704226061787082821046821,
    0.035262412660156681033782717998428,
    0.036076989622888701185500318003895,
    0.036698604498456094498018047441094,
    0.037120549269832576114119958413599,
    0.037334228751935040321235449094698,
    0.037361073762679023410321241766599,  # centre
]


def _rescale_error(
    err: Float64, result_abs: Float64, result_asc: Float64
) -> Float64:
    var e = abs(err)
    if result_asc != 0.0 and e != 0.0:
        var scale = pow(200.0 * e / result_asc, 1.5)
        e = result_asc * scale if scale < 1.0 else result_asc
    if result_abs > MSL_DBL_MIN / (50.0 * MSL_DBL_EPSILON):
        var min_err = 50.0 * MSL_DBL_EPSILON * result_abs
        if min_err > e:
            e = min_err
    return e


def qng_integrate[
    fn_: def(Float64) capturing -> Float64
](
    a: Float64, b: Float64, epsabs: Float64, epsrel: Float64
) -> IntegrationResult:
    """Non-adaptive Gauss-Kronrod integration, up to 87-point rule.

    Tries progressively higher-order rules (21 → 43 → 87 points), reusing
    function evaluations, until the tolerance is satisfied.

    Parameters:
        fn_: Scalar function to integrate.

    Args:
        a: Lower limit.
        b: Upper limit.
        epsabs: Absolute error tolerance.
        epsrel: Relative error tolerance.

    Returns:
        IntegrationResult with val (integral) and err (error estimate).
    """
    var result = IntegrationResult()

    if epsabs <= 0.0 and (epsrel < 50.0 * MSL_DBL_EPSILON or epsrel < 0.5e-28):
        return result^

    var half_length = 0.5 * (b - a)
    var abs_half_length = abs(half_length)
    var center = 0.5 * (b + a)
    var f_center = fn_(center)

    # savfun stores paired sums f(c+x)+f(c-x) for all 21 non-centre points
    var savfun = Array[Float64, 21](uninitialized=True)
    var fv1 = Array[Float64, 5](uninitialized=True)
    var fv2 = Array[Float64, 5](uninitialized=True)
    var fv3 = Array[Float64, 5](uninitialized=True)
    var fv4 = Array[Float64, 5](uninitialized=True)

    # ===--- Level 1: 10-point Gauss / 21-point Kronrod ---===
    var res10: Float64 = 0.0
    var res21: Float64 = w21b[5] * f_center
    var resabs: Float64 = w21b[5] * abs(f_center)

    for k in range(5):
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

    for k in range(5):
        var abscissa = half_length * x2[k]
        var fval1 = fn_(center + abscissa)
        var fval2 = fn_(center - abscissa)
        var fval = fval1 + fval2
        res21 += w21b[k] * fval
        resabs += w21b[k] * (abs(fval1) + abs(fval2))
        savfun[k + 5] = fval
        fv3[k] = fval1
        fv4[k] = fval2

    resabs *= abs_half_length

    var mean = 0.5 * res21
    var resasc = w21b[5] * abs(f_center - mean)
    for k in range(5):
        resasc += w21a[k] * (abs(fv1[k] - mean) + abs(fv2[k] - mean))
        resasc += w21b[k] * (abs(fv3[k] - mean) + abs(fv4[k] - mean))
    resasc *= abs_half_length

    var err21 = _rescale_error((res21 - res10) * half_length, resabs, resasc)

    if err21 <= max(epsabs, epsrel * abs(res21 * half_length)):
        result.val = res21 * half_length
        result.err = err21
        return result^

    # ===--- Level 2: 43-point Kronrod extension ---===
    var res43: Float64 = w43b[11] * f_center
    for k in range(10):
        res43 += savfun[k] * w43a[k]

    var idx: Int = 10
    for k in range(11):
        var abscissa = half_length * x3[k]
        var fval = fn_(center + abscissa) + fn_(center - abscissa)
        res43 += w43b[k] * fval
        savfun[idx] = fval
        idx += 1

    var err43 = _rescale_error((res43 - res21) * half_length, resabs, resasc)

    if err43 <= max(epsabs, epsrel * abs(res43 * half_length)):
        result.val = res43 * half_length
        result.err = err43
        return result^

    # ===--- Level 3: 87-point Kronrod extension ---===
    var res87: Float64 = w87b[22] * f_center
    for k in range(21):
        res87 += savfun[k] * w87a[k]

    for k in range(22):
        var abscissa = half_length * x4[k]
        res87 += w87b[k] * (fn_(center + abscissa) + fn_(center - abscissa))

    var err87 = _rescale_error((res87 - res43) * half_length, resabs, resasc)

    result.val = res87 * half_length
    result.err = err87
    return result^
