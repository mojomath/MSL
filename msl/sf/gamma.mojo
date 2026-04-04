# SPDX-License-Identifier: GPL-3.0-or-later

# ===----------------------------------------------------------------------=== #
# MSL (Mojo Scientific Library)
#
# Derived from GNU Scientific Library (GSL)
# Original file: specfunc/gamma.c
#
# Original authors:
# Copyright (C) 1996-2000 Gerard Jungman
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
Gamma functions.

This is a direct port of the GSL special functions implementation.
"""

from msl.core.const import MSL_DBL_EPSILON, MSL_SQRT_DBL_EPSILON, MSL_PI
from msl.sf.result import SFSResult
from std.math import sqrt, exp, log, cos, sin, abs, floor
from std.collections import InlineArray


comptime LogRootTwoPi: Float64 = 0.9189385332046727418


# ===----------------------------------------------------------------------=== #
# Lanczos coefficients for gamma=7, kmax=8
# ===----------------------------------------------------------------------=== #

comptime lanczos_c: InlineArray[Float64, 9] = [
    0.99999999999980993227684700473478,
    676.520368121885098567009190444019,
    -1259.13921672240287047156078755283,
    771.3234287776530788486528258894,
    -176.61502916214059906584551354,
    12.507343278686904814458936853,
    -0.13857109526572011689554707,
    9.984369578019570859563e-6,
    1.50563273514931155834e-7,
]


# ===----------------------------------------------------------------------=== #
# Precomputed factorials for n = 0 to n = 150 (151 values)
# GSL uses 170 for fact, 297 for doublefact but we limit both to 150
# because Mojo cannot compile large InlineArray literals (>200 elements)
# ===----------------------------------------------------------------------=== #

comptime fact_table: InlineArray[Float64, 151] = [
    1.00000000000e+00,
    1.00000000000e+00,
    2.00000000000e+00,
    6.00000000000e+00,
    2.40000000000e+01,
    1.20000000000e+02,
    7.20000000000e+02,
    5.04000000000e+03,
    4.03200000000e+04,
    3.62880000000e+05,
    3.62880000000e+06,
    3.99168000000e+07,
    4.79001600000e+08,
    6.22702080000e+09,
    8.71782912000e+10,
    1.30767436800e+12,
    2.09227898880e+13,
    3.55687428096e+14,
    6.40237370573e+15,
    1.21645100409e+17,
    2.43290200818e+18,
    5.10909421717e+19,
    1.12400072778e+21,
    2.58520167389e+22,
    6.20448401733e+23,
    1.55112100433e+25,
    4.03291461127e+26,
    1.08888694504e+28,
    3.04888344612e+29,
    8.84176199374e+30,
    2.65252859812e+32,
    8.22283865418e+33,
    2.63130836934e+35,
    8.68331761881e+36,
    2.95232799040e+38,
    1.03331479664e+40,
    3.71993326790e+41,
    1.37637530912e+43,
    5.23022617467e+44,
    2.03978820812e+46,
    8.15915283248e+47,
    3.34525266132e+49,
    1.40500611775e+51,
    6.04152630634e+52,
    2.65827157479e+54,
    1.19622220865e+56,
    5.50262215981e+57,
    2.58623241511e+59,
    1.24139155925e+61,
    6.08281864034e+62,
    3.04140932017e+64,
    1.55111875329e+66,
    8.06581751709e+67,
    4.27488328406e+69,
    2.30843697339e+71,
    1.26964033537e+73,
    7.10998587805e+74,
    4.05269195049e+76,
    2.35056133128e+78,
    1.38683118546e+80,
    8.32098711274e+81,
    5.07580213877e+83,
    3.14699732604e+85,
    1.98260831540e+87,
    1.26886932186e+89,
    8.24765059208e+90,
    5.44344939077e+92,
    3.64711109182e+94,
    2.48003554244e+96,
    1.71122452428e+98,
    1.19785716700e+100,
    8.50478588568e+101,
    6.12344583769e+103,
    4.47011546151e+105,
    3.30788544152e+107,
    2.48091408114e+109,
    1.88549470167e+111,
    1.45183092028e+113,
    1.13242811782e+115,
    8.94618213078e+116,
    7.15694570463e+118,
    5.79712602075e+120,
    4.75364333701e+122,
    3.94552396972e+124,
    3.31424013457e+126,
    2.81710411438e+128,
    2.42270953837e+130,
    2.10775729838e+132,
    1.85482642257e+134,
    1.65079551609e+136,
    1.48571596448e+138,
    1.35200152768e+140,
    1.24384140546e+142,
    1.15677250708e+144,
    1.08736615666e+146,
    1.03299784882e+148,
    9.91677934871e+149,
    9.61927596825e+151,
    9.42689044888e+153,
    9.33262154439e+155,
    9.33262154439e+157,
    9.42594775984e+159,
    9.61446671504e+161,
    9.90290071649e+163,
    1.02990167451e+166,
    1.08139675824e+168,
    1.14628056373e+170,
    1.22652020320e+172,
    1.32464181945e+174,
    1.44385958320e+176,
    1.58824554152e+178,
    1.76295255109e+180,
    1.97450685722e+182,
    2.23119274866e+184,
    2.54355973347e+186,
    2.92509369349e+188,
    3.39310868445e+190,
    3.96993716081e+192,
    4.68452584975e+194,
    5.57458576121e+196,
    6.68950291345e+198,
    8.09429852527e+200,
    9.87504420083e+202,
    1.21463043670e+205,
    1.50614174151e+207,
    1.88267717689e+209,
    2.37217324288e+211,
    3.01266001846e+213,
    3.85620482363e+215,
    4.97450422248e+217,
    6.46685548922e+219,
    8.47158069088e+221,
    1.11824865120e+224,
    1.48727070609e+226,
    1.99294274616e+228,
    2.69047270732e+230,
    3.65904288195e+232,
    5.01288874827e+234,
    6.91778647262e+236,
    9.61572319694e+238,
    1.34620124757e+241,
    1.89814375908e+243,
    2.69536413789e+245,
    3.85437071718e+247,
    5.55029383274e+249,
    8.04792605747e+251,
    1.17499720439e+254,
    1.72724589045e+256,
    2.55632391787e+258,
    3.80892263763e+260,
    5.71338395645e+262,
]


# ===----------------------------------------------------------------------=== #
# Precomputed double factorials for n = 0 to n = 150 (151 values)
# n!! = n * (n-2) * (n-4) * ... (ends at 1 or 2)
# ===----------------------------------------------------------------------=== #

comptime doublefact_table: InlineArray[Float64, 151] = [
    1.00000000000e+00,
    1.00000000000e+00,
    2.00000000000e+00,
    3.00000000000e+00,
    8.00000000000e+00,
    1.50000000000e+01,
    4.80000000000e+01,
    1.05000000000e+02,
    3.84000000000e+02,
    9.45000000000e+02,
    3.84000000000e+03,
    1.03950000000e+04,
    4.60800000000e+04,
    1.35135000000e+05,
    6.45120000000e+05,
    2.02702500000e+06,
    1.03219200000e+07,
    3.44594250000e+07,
    1.85794560000e+08,
    6.54729075000e+08,
    3.71589120000e+09,
    1.37493105750e+10,
    8.17496064000e+10,
    3.16234143225e+11,
    1.96199055360e+12,
    7.90585358062e+12,
    5.10117543936e+13,
    2.13458046677e+14,
    1.42832912302e+15,
    6.19028335363e+15,
    4.28498736906e+16,
    1.91898783963e+17,
    1.37119595810e+18,
    6.33265987076e+18,
    4.66206625754e+19,
    2.21643095477e+20,
    1.67834385271e+21,
    8.20079453264e+21,
    6.37770664031e+22,
    3.19830986773e+23,
    2.55108265613e+24,
    1.31130704577e+25,
    1.07145471557e+26,
    5.63862029681e+26,
    4.71440074852e+27,
    2.53737913356e+28,
    2.16862434432e+29,
    1.19256819277e+30,
    1.04093968527e+31,
    5.84358414459e+31,
    5.20469842637e+32,
    2.98022791374e+33,
    2.70644318171e+34,
    1.57952079428e+35,
    1.46147931812e+36,
    8.68736436856e+36,
    8.18428418149e+37,
    4.95179769008e+38,
    4.74688482527e+39,
    2.92156063715e+40,
    2.84813089516e+41,
    1.78215198866e+42,
    1.76584115500e+43,
    1.12275575286e+44,
    1.13013833920e+45,
    7.29791239356e+45,
    7.45891303872e+46,
    4.88960130369e+47,
    5.07206086633e+48,
    3.37382489954e+49,
    3.55044260643e+50,
    2.39541567868e+51,
    2.55631867663e+52,
    1.74865344543e+53,
    1.89167582071e+54,
    1.31149008408e+55,
    1.43767362374e+56,
    1.00984736474e+57,
    1.12138542651e+58,
    7.97779418143e+58,
    8.97108341211e+59,
    6.46201328696e+60,
    7.35628839793e+61,
    5.36347102817e+62,
    6.17928225426e+63,
    4.55895037395e+64,
    5.31418273867e+65,
    3.96628682534e+66,
    4.67648081003e+67,
    3.52999527455e+68,
    4.20883272902e+69,
    3.21229569984e+70,
    3.87212611070e+71,
    2.98743500085e+72,
    3.63979854406e+73,
    2.83806325081e+74,
    3.49420660230e+75,
    2.75292135328e+76,
    3.42432247025e+77,
    2.72539213975e+78,
    3.42432247025e+79,
    2.75264606115e+80,
    3.49280891966e+81,
    2.83522544298e+82,
    3.63252127644e+83,
    2.97698671513e+84,
    3.85047255303e+85,
    3.18537578519e+86,
    4.15851035727e+87,
    3.47205960586e+88,
    4.57436139300e+89,
    3.85398616250e+90,
    5.12328476016e+91,
    4.35500436363e+92,
    5.84054462658e+93,
    5.00825501817e+94,
    6.77503176683e+95,
    5.85965837126e+96,
    7.99453748486e+97,
    6.97299346180e+98,
    9.59344498184e+99,
    8.43732208878e+100,
    1.17040028778e+102,
    1.03779061692e+103,
    1.45129635685e+104,
    1.29723827115e+105,
    1.82863340963e+106,
    1.64749260436e+107,
    2.34065076433e+108,
    2.12526545962e+109,
    3.04284599363e+110,
    2.78409775211e+111,
    4.01655671159e+112,
    3.70285001030e+113,
    5.38218599353e+114,
    4.99884751391e+115,
    7.31977295121e+116,
    6.84842109406e+117,
    1.01012866727e+119,
    9.51930532074e+119,
    1.41418013417e+121,
    1.34222205022e+122,
    2.00813579053e+123,
    1.91937753182e+124,
    2.89171553836e+125,
    2.78309742114e+126,
    4.22190468600e+127,
    4.09115320908e+128,
    6.24841893528e+129,
    6.09581828152e+130,
    9.37262840292e+131,
]


# ===----------------------------------------------------------------------=== #
# Lanczos method for ln(Gamma(x)) with x > 0
# ===----------------------------------------------------------------------=== #

def lngamma_lanczos(x: Float64) -> Float64:
    var x_adj = x - 1.0

    var Ag = lanczos_c[0]
    var k = 1
    while k <= 8:
        Ag += lanczos_c[k] / (x_adj + Float64(k))
        k += 1

    var term1 = (x_adj + 0.5) * log((x_adj + 7.5) / 2.718281828459045)
    var term2 = LogRootTwoPi + log(Ag)
    return term1 + (term2 - 7.0)


# ===----------------------------------------------------------------------=== #
# Log Gamma
# ===----------------------------------------------------------------------=== #

def _lngamma(x: Float64) -> SFSResult:
    var result = SFSResult()

    if abs(x - 1.0) < 0.01:
        var z = x - 1.0
        var p: Float64 = 0.08106146479212740
        p = p + z * 0.02360195206550424
        p = p + z * z * 0.01797061784304505
        p = p + z * z * z * 0.00661637535186565
        var q: Float64 = 1.0
        q = q + z * 1.32132397893142
        q = q + z * z * 0.991059544157268
        q = q + z * z * z * 0.273084002288770
        result.val = z * (p / q)
        result.err = MSL_DBL_EPSILON * abs(result.val) / (MSL_DBL_EPSILON + abs(z))
        return result^

    elif abs(x - 2.0) < 0.01:
        var z = x - 2.0
        var p: Float64 = -0.01355099545459314
        p = p + z * 0.05426485116219510
        p = p + z * z * 0.20635748813559287
        p = p + z * z * z * 0.26596683588745889
        p = p + z * z * z * z * 0.08233196972957289
        var q: Float64 = 1.0
        q = q + z * 1.26104139941502
        q = q + z * z * 0.986647372772498
        q = q + z * z * z * 0.268380494557498
        result.val = z * (p / q)
        result.err = MSL_DBL_EPSILON * abs(result.val) / (MSL_DBL_EPSILON + abs(z))
        return result^

    elif x >= 0.5:
        result.val = lngamma_lanczos(x)
        result.err = 2.0 * MSL_DBL_EPSILON * abs(result.val)
        return result^

    elif x == 0.0:
        result.val = 0.0
        result.err = 0.0
        return result^

    elif abs(x) < 0.02:
        var c1 = -0.07721566490153286061
        var c2 = -0.01094400467202744461
        var c3 = 0.09252092391911371098
        var c4 = -0.01827191316559981266
        var c5 = 0.01800493109685479790
        var c6 = -0.00685088537872380685
        var c7 = 0.00399823955756846603
        var c8 = -0.00189430621687107802

        var eps = x
        var eps2 = eps * eps
        var ser = 1.0 + eps * (c1 + eps * (c2 + eps * (c3 + eps * (c4 + eps * (c5 + eps * (c6 + eps * (c7 + eps * c8)))))))
        result.val = log(abs(eps)) + c1 * eps + 0.5 * c2 * eps2 + c3 / 3.0 * eps2 * eps + c4 / 4.0 * eps2 * eps2 + c5 / 5.0 * eps2 * eps2 * eps + c6 / 6.0 * eps2 * eps2 * eps2 + c7 / 7.0 * eps2 * eps2 * eps2 * eps + c8 / 8.0 * eps2 * eps2 * eps2 * eps2 - log(ser)

        var lna = log(abs(eps))
        result.err = 2.0 * MSL_DBL_EPSILON * (abs(lna) + abs(result.val))
        return result^

    else:
        var z = 1.0 - x
        var s = sin(MSL_PI * z)
        var as_val = abs(s)

        if s == 0.0:
            result.val = 0.0
            result.err = 0.0
            return result^

        var lg_z = lngamma_lanczos(z)
        result.val = log(MSL_PI) - (log(as_val) + lg_z)
        result.err = 2.0 * MSL_DBL_EPSILON * abs(result.val) + lg_z
        return result^


# ===----------------------------------------------------------------------=== #
# Gamma function
# ===----------------------------------------------------------------------=== #

def _gamma(x: Float64) -> SFSResult:
    var result = SFSResult()

    if x <= 0.0:
        result.val = 0.0
        result.err = 0.0
        return result^

    if x < 0.5:
        var rint_x = Int(floor(x + 0.5))
        var f_x = x - Float64(rint_x)
        var sgn_gamma = 1.0 if (rint_x % 2) == 0 else -1.0
        var sin_term = sgn_gamma * sin(MSL_PI * f_x) / MSL_PI

        if sin_term == 0.0:
            result.val = 0.0
            result.err = 0.0
            return result^

        var g = _gamma_lanczos(1.0 - x)
        result.val = 1.0 / (sin_term * g.val)
        result.err = abs(g.err / g.val) * abs(result.val) + 2.0 * MSL_DBL_EPSILON * abs(result.val)
        return result^

    return _gamma_lanczos(x)


def _gamma_lanczos(x: Float64) -> SFSResult:
    var result = SFSResult()

    result.val = lngamma_lanczos(x)
    result.val = exp(result.val)

    result.err = MSL_DBL_EPSILON * abs(result.val)
    return result^


# ===----------------------------------------------------------------------=== #
# Regulated Gamma function: Gamma*(x)
# ===----------------------------------------------------------------------=== #

def _gammastar(x: Float64) -> SFSResult:
    var result = SFSResult()

    if x <= 0.0:
        result.val = 0.0
        result.err = 0.0
        return result^

    if x < 0.5:
        var lg = lngamma(x)
        var lx = log(x)
        var c = 0.5 * (log(2.0) + log(MSL_PI))
        var lnr_val = lg.val - (x - 0.5) * lx + x - c
        result.val = exp(lnr_val)
        result.err = lg.err * result.val + 2.0 * MSL_DBL_EPSILON * abs(result.val)
        return result^

    if x < 2.0:
        var t = 4.0 / 3.0 * (x - 0.5) - 1.0

        var gstar_cs: InlineArray[Float64, 13] = [
            0.00250662827563475527079,
            0.12431359316440235670,
            0.03813225933718515534,
            0.01517487570252815878,
            0.00641024341885595294,
            0.00122725581439983491,
            0.00048052551446631236,
            0.00018445529376203228,
            0.00006920154062104890,
            0.00002889355840717814,
            0.00001284226326359862,
            0.00000499999999999998,
            0.00000000000000000000,
        ]

        result = cheb_eval(gstar_cs, 12, -1.0, 1.0, t)
        return result^

    if x < 10.0:
        var t = 0.25 * (x - 2.0) - 1.0

        var gstar_cs2: InlineArray[Float64, 13] = [
            0.00000352962225500422260,
            0.00013059433214083752768,
            0.00104780268165977166,
            0.00476468413548075608,
            0.01075483845476259620,
            0.01477275350451643210,
            0.01323141007828090980,
            0.00960766099356476296,
            0.00536361025254409050,
            0.00230683521611475210,
            0.00077839119658395287,
            0.00021592992038751293,
            0.00005443760491892778,
        ]

        var c = cheb_eval(gstar_cs2, 12, -1.0, 1.0, t)
        result.val = c.val / (x * x) + 1.0 + 1.0 / (12.0 * x)
        result.err = c.err / (x * x) + 2.0 * MSL_DBL_EPSILON * abs(result.val)
        return result^

    result.val = 1.0 + 1.0 / (12.0 * x) + 1.0 / (288.0 * x * x)
    result.err = MSL_DBL_EPSILON * abs(result.val)
    return result^


# ===----------------------------------------------------------------------=== #
# Chebyshev series evaluation
# ===----------------------------------------------------------------------=== #

def cheb_eval[N: Int](
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
# 1/Gamma(x) - reciprocal gamma
# ===----------------------------------------------------------------------=== #

def _gammainv(x: Float64) -> SFSResult:
    var result = SFSResult()

    if x <= 0.0:
        result.val = 0.0
        result.err = 0.0
        return result^

    var lg = lngamma(x)
    result.val = exp(-lg.val)
    result.err = result.val * lg.err
    result.err += MSL_DBL_EPSILON * abs(result.val)
    return result^


# ===----------------------------------------------------------------------=== #
# Factorial n!
# ===----------------------------------------------------------------------=== #

def fact(n: UInt64) -> SFSResult:
    var result = SFSResult()

    if n > 150:
        result.val = 0.0
        result.err = 0.0
        return result^

    result.val = fact_table[n]
    result.err = MSL_DBL_EPSILON * abs(result.val)
    return result^


# ===----------------------------------------------------------------------=== #
# Double factorial n!!
# ===----------------------------------------------------------------------=== #

def doublefact(n: UInt64) -> SFSResult:
    var result = SFSResult()

    if n > 150:
        result.val = 0.0
        result.err = 0.0
        return result^

    result.val = doublefact_table[n]
    result.err = MSL_DBL_EPSILON * abs(result.val)
    return result^


# ===----------------------------------------------------------------------=== #
# Log factorial log(n!)
# ===----------------------------------------------------------------------=== #

def lnfact(n: UInt64) -> SFSResult:
    if n <= 150:
        var result = SFSResult()
        result.val = log(fact_table[n])
        result.err = MSL_DBL_EPSILON * abs(result.val)
        return result^
    
    return _lngamma(Float64(n) + 1.0)


# ===----------------------------------------------------------------------=== #
# Log double factorial log(n!!)
# ===----------------------------------------------------------------------=== #

def lndoublefact(n: UInt64) -> SFSResult:
    if n <= 150:
        var result = SFSResult()
        result.val = log(doublefact_table[n])
        result.err = MSL_DBL_EPSILON * abs(result.val)
        return result^
    
    var result = SFSResult()
    if n % 2 == 1:
        var lg = _lngamma(0.5 * Float64(n + 2))
        result.val = 0.5 * Float64(n + 1) * log(2.0) - 0.5 * log(MSL_PI) + lg.val
    else:
        var lg = _lngamma(0.5 * Float64(n) + 1.0)
        result.val = 0.5 * Float64(n) * log(2.0) + lg.val
    
    result.err = 2.0 * MSL_DBL_EPSILON * abs(result.val)
    return result^


# ===----------------------------------------------------------------------=== #
# Beta function
# ===----------------------------------------------------------------------=== #

def _beta(a: Float64, b: Float64) -> SFSResult:
    var result = SFSResult()

    if a <= 0.0 or b <= 0.0:
        result.val = 0.0
        result.err = 0.0
        return result^

    if a > b:
        return _beta(b, a)

    if a + b < 170.0:
        result.val = exp(lngamma(a + b).val - lngamma(a).val - lngamma(b).val)
        result.err = result.val * (lngamma(a + b).err + lngamma(a).err + lngamma(b).err)
    else:
        var lg_a = lngamma(a)
        var lg_b = lngamma(b)
        var lg_ab = lngamma(a + b)

        if lg_ab.val - lg_a.val - lg_b.val < -700.0:
            result.val = 0.0
            result.err = MSL_DBL_EPSILON
        else:
            result.val = exp(lg_ab.val - lg_a.val - lg_b.val)
            result.err = result.val * (lg_ab.err + lg_a.err + lg_b.err)

    result.err += MSL_DBL_EPSILON * abs(result.val)
    return result^


# ===----------------------------------------------------------------------=== #
# Log Beta function
# ===----------------------------------------------------------------------=== #

def _lnbeta(a: Float64, b: Float64) -> SFSResult:
    var result = SFSResult()

    if a <= 0.0 or b <= 0.0:
        result.val = 0.0
        result.err = 0.0
        return result^

    var lg_a = lngamma(a)
    var lg_b = lngamma(b)
    var lg_ab = lngamma(a + b)

    result.val = lg_ab.val - lg_a.val - lg_b.val
    result.err = lg_ab.err + lg_a.err + lg_b.err
    result.err += MSL_DBL_EPSILON * abs(result.val)
    return result^


# ===----------------------------------------------------------------------=== #
# Public API
# ===----------------------------------------------------------------------=== #

def gamma(x: Float64) -> SFSResult:
    if x <= 0.0:
        var result = SFSResult()
        result.val = 0.0
        result.err = 0.0
        return result^

    if x < 0.5:
        var rint_x = Int(floor(x + 0.5))
        var f_x = x - Float64(rint_x)
        var sgn_gamma = 1.0 if (rint_x % 2) == 0 else -1.0
        var sin_term = sgn_gamma * sin(MSL_PI * f_x) / MSL_PI

        if sin_term == 0.0:
            var result = SFSResult()
            result.val = 0.0
            result.err = 0.0
            return result^

        var g = _gamma_lanczos(1.0 - x)
        var result = SFSResult()
        result.val = 1.0 / (sin_term * g.val)
        result.err = abs(g.err / g.val) * abs(result.val) + 2.0 * MSL_DBL_EPSILON * abs(result.val)
        return result^

    return _gamma_lanczos(x)


def lngamma(x: Float64) -> SFSResult:
    return _lngamma(x)


def gammastar(x: Float64) -> SFSResult:
    return _gammastar(x)


def gammainv(x: Float64) -> SFSResult:
    return _gammainv(x)


def factorial(n: UInt64) -> SFSResult:
    return fact(n)


def double_factorial(n: UInt64) -> SFSResult:
    return doublefact(n)


def ln_factorial(n: UInt64) -> SFSResult:
    return lnfact(n)


def ln_double_factorial(n: UInt64) -> SFSResult:
    return lndoublefact(n)


def beta(a: Float64, b: Float64) -> SFSResult:
    return _beta(a, b)


def lnbeta(a: Float64, b: Float64) -> SFSResult:
    return _lnbeta(a, b)
