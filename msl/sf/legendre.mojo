# SPDX-License-Identifier: GPL-3.0-or-later

# ===----------------------------------------------------------------------=== #
# MSL (Mojo Scientific Library)
#
# Derived from GNU Scientific Library (GSL)
# Original file: specfunc/legendre_poly.c
#
# Original authors:
# Copyright (C) 1996-2002 Gerard Jungman
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
Legendre polynomials and functions.

This is a direct port of the GSL special functions implementation.
"""

from msl.core.const import MSL_DBL_EPSILON, MSL_SQRT_DBL_EPSILON, MSL_PI
from msl.sf.result import SFSResult
from msl.sf.gamma import lngamma
from std.math import sqrt, log, cos, sin, acos, abs, floor
from msl.sf.bessel import bessel_J0, bessel_Jn, bessel_Jnu_asympx


def _legendre_P1(x: Float64) -> SFSResult:
    var result = SFSResult()
    result.val = x
    result.err = 0.0
    return result^


def _legendre_P2(x: Float64) -> SFSResult:
    var result = SFSResult()
    result.val = 0.5 * (3.0 * x * x - 1.0)
    result.err = MSL_DBL_EPSILON * (abs(3.0 * x * x) + 1.0)
    return result^


def _legendre_P3(x: Float64) -> SFSResult:
    var result = SFSResult()
    result.val = 0.5 * x * (5.0 * x * x - 3.0)
    result.err = MSL_DBL_EPSILON * (abs(result.val) + 0.5 * abs(x) * (abs(5.0 * x * x) + 3.0))
    return result^


def legendre_Pl(l: Int, x: Float64) -> SFSResult:
    if l < 0:
        var result = SFSResult()
        result.val = 0.0
        result.err = 0.0
        return result^
    
    if x < -1.0 or x > 1.0:
        var result = SFSResult()
        result.val = 0.0
        result.err = 0.0
        return result^
    
    if l == 0:
        var result = SFSResult()
        result.val = 1.0
        result.err = 0.0
        return result^
    
    if l == 1:
        return _legendre_P1(x)
    
    if l == 2:
        return _legendre_P2(x)
    
    if l == 3:
        return _legendre_P3(x)
    
    if x == 1.0:
        var result = SFSResult()
        result.val = 1.0
        result.err = 0.0
        return result^
    
    if x == -1.0:
        var result = SFSResult()
        if l % 2 == 1:
            result.val = -1.0
        else:
            result.val = 1.0
        result.err = 0.0
        return result^
    
    if l < 100000:
        var p_ellm2: Float64 = 1.0
        var p_ellm1: Float64 = x
        var p_ell: Float64 = p_ellm1

        var e_ellm2: Float64 = MSL_DBL_EPSILON
        var e_ellm1: Float64 = abs(x) * MSL_DBL_EPSILON
        var e_ell: Float64 = e_ellm1

        var ell = 2
        while ell <= l:
            p_ell = (x * Float64(2 * ell - 1) * p_ellm1 - Float64(ell - 1) * p_ellm2) / Float64(ell)
            p_ellm2 = p_ellm1
            p_ellm1 = p_ell

            e_ell = 0.5 * (abs(x) * Float64(2 * ell - 1) * e_ellm1 + Float64(ell - 1) * e_ellm2) / Float64(ell)
            e_ellm2 = e_ellm1
            e_ellm1 = e_ell
            ell += 1

        var result = SFSResult()
        result.val = p_ell
        result.err = e_ell + Float64(l) * abs(p_ell) * MSL_DBL_EPSILON
        return result^
    
    var u = Float64(l) + 0.5
    var th = acos(x)
    var j0 = bessel_Jnu_asympx(u, u * th)
    var jm1 = bessel_Jnu_asympx(u, u * th)
    
    var pre: Float64
    var B00: Float64
    
    if th < MSL_SQRT_DBL_EPSILON:
        B00 = (1.0 + th * th / 15.0) / 24.0
        pre = 1.0 + th * th / 12.0
    else:
        var sin_th = sqrt(1.0 - x * x)
        var cot_th = x / sin_th
        B00 = 1.0 / 8.0 * (1.0 - th * cot_th) / (th * th)
        pre = sqrt(th / sin_th)

    var c1 = th / u * B00

    var result = SFSResult()
    result.val = pre * (j0.val + c1 * jm1.val)
    result.err = pre * (j0.err + abs(c1) * jm1.err)
    result.err += MSL_SQRT_DBL_EPSILON * abs(result.val)
    return result^


def legendre_P1(x: Float64) -> SFSResult:
    return _legendre_P1(x)


def legendre_P2(x: Float64) -> SFSResult:
    return _legendre_P2(x)


def legendre_P3(x: Float64) -> SFSResult:
    return _legendre_P3(x)
