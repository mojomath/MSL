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
    1.00000000000e00,
    1.00000000000e00,
    2.00000000000e00,
    6.00000000000e00,
    2.40000000000e01,
    1.20000000000e02,
    7.20000000000e02,
    5.04000000000e03,
    4.03200000000e04,
    3.62880000000e05,
    3.62880000000e06,
    3.99168000000e07,
    4.79001600000e08,
    6.22702080000e09,
    8.71782912000e10,
    1.30767436800e12,
    2.09227898880e13,
    3.55687428096e14,
    6.40237370573e15,
    1.21645100409e17,
    2.43290200818e18,
    5.10909421717e19,
    1.12400072778e21,
    2.58520167389e22,
    6.20448401733e23,
    1.55112100433e25,
    4.03291461127e26,
    1.08888694504e28,
    3.04888344612e29,
    8.84176199374e30,
    2.65252859812e32,
    8.22283865418e33,
    2.63130836934e35,
    8.68331761881e36,
    2.95232799040e38,
    1.03331479664e40,
    3.71993326790e41,
    1.37637530912e43,
    5.23022617467e44,
    2.03978820812e46,
    8.15915283248e47,
    3.34525266132e49,
    1.40500611775e51,
    6.04152630634e52,
    2.65827157479e54,
    1.19622220865e56,
    5.50262215981e57,
    2.58623241511e59,
    1.24139155925e61,
    6.08281864034e62,
    3.04140932017e64,
    1.55111875329e66,
    8.06581751709e67,
    4.27488328406e69,
    2.30843697339e71,
    1.26964033537e73,
    7.10998587805e74,
    4.05269195049e76,
    2.35056133128e78,
    1.38683118546e80,
    8.32098711274e81,
    5.07580213877e83,
    3.14699732604e85,
    1.98260831540e87,
    1.26886932186e89,
    8.24765059208e90,
    5.44344939077e92,
    3.64711109182e94,
    2.48003554244e96,
    1.71122452428e98,
    1.19785716700e100,
    8.50478588568e101,
    6.12344583769e103,
    4.47011546151e105,
    3.30788544152e107,
    2.48091408114e109,
    1.88549470167e111,
    1.45183092028e113,
    1.13242811782e115,
    8.94618213078e116,
    7.15694570463e118,
    5.79712602075e120,
    4.75364333701e122,
    3.94552396972e124,
    3.31424013457e126,
    2.81710411438e128,
    2.42270953837e130,
    2.10775729838e132,
    1.85482642257e134,
    1.65079551609e136,
    1.48571596448e138,
    1.35200152768e140,
    1.24384140546e142,
    1.15677250708e144,
    1.08736615666e146,
    1.03299784882e148,
    9.91677934871e149,
    9.61927596825e151,
    9.42689044888e153,
    9.33262154439e155,
    9.33262154439e157,
    9.42594775984e159,
    9.61446671504e161,
    9.90290071649e163,
    1.02990167451e166,
    1.08139675824e168,
    1.14628056373e170,
    1.22652020320e172,
    1.32464181945e174,
    1.44385958320e176,
    1.58824554152e178,
    1.76295255109e180,
    1.97450685722e182,
    2.23119274866e184,
    2.54355973347e186,
    2.92509369349e188,
    3.39310868445e190,
    3.96993716081e192,
    4.68452584975e194,
    5.57458576121e196,
    6.68950291345e198,
    8.09429852527e200,
    9.87504420083e202,
    1.21463043670e205,
    1.50614174151e207,
    1.88267717689e209,
    2.37217324288e211,
    3.01266001846e213,
    3.85620482363e215,
    4.97450422248e217,
    6.46685548922e219,
    8.47158069088e221,
    1.11824865120e224,
    1.48727070609e226,
    1.99294274616e228,
    2.69047270732e230,
    3.65904288195e232,
    5.01288874827e234,
    6.91778647262e236,
    9.61572319694e238,
    1.34620124757e241,
    1.89814375908e243,
    2.69536413789e245,
    3.85437071718e247,
    5.55029383274e249,
    8.04792605747e251,
    1.17499720439e254,
    1.72724589045e256,
    2.55632391787e258,
    3.80892263763e260,
    5.71338395645e262,
]


# ===----------------------------------------------------------------------=== #
# Precomputed double factorials for n = 0 to n = 150 (151 values)
# n!! = n * (n-2) * (n-4) * ... (ends at 1 or 2)
# ===----------------------------------------------------------------------=== #

comptime doublefact_table: InlineArray[Float64, 151] = [
    1.00000000000e00,
    1.00000000000e00,
    2.00000000000e00,
    3.00000000000e00,
    8.00000000000e00,
    1.50000000000e01,
    4.80000000000e01,
    1.05000000000e02,
    3.84000000000e02,
    9.45000000000e02,
    3.84000000000e03,
    1.03950000000e04,
    4.60800000000e04,
    1.35135000000e05,
    6.45120000000e05,
    2.02702500000e06,
    1.03219200000e07,
    3.44594250000e07,
    1.85794560000e08,
    6.54729075000e08,
    3.71589120000e09,
    1.37493105750e10,
    8.17496064000e10,
    3.16234143225e11,
    1.96199055360e12,
    7.90585358062e12,
    5.10117543936e13,
    2.13458046677e14,
    1.42832912302e15,
    6.19028335363e15,
    4.28498736906e16,
    1.91898783963e17,
    1.37119595810e18,
    6.33265987076e18,
    4.66206625754e19,
    2.21643095477e20,
    1.67834385271e21,
    8.20079453264e21,
    6.37770664031e22,
    3.19830986773e23,
    2.55108265613e24,
    1.31130704577e25,
    1.07145471557e26,
    5.63862029681e26,
    4.71440074852e27,
    2.53737913356e28,
    2.16862434432e29,
    1.19256819277e30,
    1.04093968527e31,
    5.84358414459e31,
    5.20469842637e32,
    2.98022791374e33,
    2.70644318171e34,
    1.57952079428e35,
    1.46147931812e36,
    8.68736436856e36,
    8.18428418149e37,
    4.95179769008e38,
    4.74688482527e39,
    2.92156063715e40,
    2.84813089516e41,
    1.78215198866e42,
    1.76584115500e43,
    1.12275575286e44,
    1.13013833920e45,
    7.29791239356e45,
    7.45891303872e46,
    4.88960130369e47,
    5.07206086633e48,
    3.37382489954e49,
    3.55044260643e50,
    2.39541567868e51,
    2.55631867663e52,
    1.74865344543e53,
    1.89167582071e54,
    1.31149008408e55,
    1.43767362374e56,
    1.00984736474e57,
    1.12138542651e58,
    7.97779418143e58,
    8.97108341211e59,
    6.46201328696e60,
    7.35628839793e61,
    5.36347102817e62,
    6.17928225426e63,
    4.55895037395e64,
    5.31418273867e65,
    3.96628682534e66,
    4.67648081003e67,
    3.52999527455e68,
    4.20883272902e69,
    3.21229569984e70,
    3.87212611070e71,
    2.98743500085e72,
    3.63979854406e73,
    2.83806325081e74,
    3.49420660230e75,
    2.75292135328e76,
    3.42432247025e77,
    2.72539213975e78,
    3.42432247025e79,
    2.75264606115e80,
    3.49280891966e81,
    2.83522544298e82,
    3.63252127644e83,
    2.97698671513e84,
    3.85047255303e85,
    3.18537578519e86,
    4.15851035727e87,
    3.47205960586e88,
    4.57436139300e89,
    3.85398616250e90,
    5.12328476016e91,
    4.35500436363e92,
    5.84054462658e93,
    5.00825501817e94,
    6.77503176683e95,
    5.85965837126e96,
    7.99453748486e97,
    6.97299346180e98,
    9.59344498184e99,
    8.43732208878e100,
    1.17040028778e102,
    1.03779061692e103,
    1.45129635685e104,
    1.29723827115e105,
    1.82863340963e106,
    1.64749260436e107,
    2.34065076433e108,
    2.12526545962e109,
    3.04284599363e110,
    2.78409775211e111,
    4.01655671159e112,
    3.70285001030e113,
    5.38218599353e114,
    4.99884751391e115,
    7.31977295121e116,
    6.84842109406e117,
    1.01012866727e119,
    9.51930532074e119,
    1.41418013417e121,
    1.34222205022e122,
    2.00813579053e123,
    1.91937753182e124,
    2.89171553836e125,
    2.78309742114e126,
    4.22190468600e127,
    4.09115320908e128,
    6.24841893528e129,
    6.09581828152e130,
    9.37262840292e131,
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
        result.err = (
            MSL_DBL_EPSILON * abs(result.val) / (MSL_DBL_EPSILON + abs(z))
        )
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
        result.err = (
            MSL_DBL_EPSILON * abs(result.val) / (MSL_DBL_EPSILON + abs(z))
        )
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
        var ser = 1.0 + eps * (
            c1
            + eps
            * (
                c2
                + eps
                * (
                    c3
                    + eps
                    * (c4 + eps * (c5 + eps * (c6 + eps * (c7 + eps * c8))))
                )
            )
        )
        result.val = (
            log(abs(eps))
            + c1 * eps
            + 0.5 * c2 * eps2
            + c3 / 3.0 * eps2 * eps
            + c4 / 4.0 * eps2 * eps2
            + c5 / 5.0 * eps2 * eps2 * eps
            + c6 / 6.0 * eps2 * eps2 * eps2
            + c7 / 7.0 * eps2 * eps2 * eps2 * eps
            + c8 / 8.0 * eps2 * eps2 * eps2 * eps2
            - log(ser)
        )

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
        result.err = abs(g.err / g.val) * abs(
            result.val
        ) + 2.0 * MSL_DBL_EPSILON * abs(result.val)
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
        result.err = lg.err * result.val + 2.0 * MSL_DBL_EPSILON * abs(
            result.val
        )
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
        result.val = (
            0.5 * Float64(n + 1) * log(2.0) - 0.5 * log(MSL_PI) + lg.val
        )
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
        result.err = result.val * (
            lngamma(a + b).err + lngamma(a).err + lngamma(b).err
        )
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
        result.err = abs(g.err / g.val) * abs(
            result.val
        ) + 2.0 * MSL_DBL_EPSILON * abs(result.val)
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
