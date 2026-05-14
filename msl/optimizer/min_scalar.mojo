# SPDX-License-Identifier: GPL-3.0-or-later

# ===----------------------------------------------------------------------=== #
# MSL (Mojo Scientific Library)
#
# Derived from GNU Scientific Library (GSL)
# Original files: min/brent.c, min/golden.c
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
# ===----------------------------------------------------------------------=== #
"""
Scalar minimization algorithms.

  min_brent   - Brent's method (parabolic interpolation + golden section)
  min_golden  - Golden section search (robust, no interpolation)

Both require a bracket (a, x, b) with f(x) < f(a) and f(x) < f(b).
"""

from std.math import abs, sqrt

from msl.core.const import MSL_DBL_EPSILON
from msl.core.errno import MSL_SUCCESS, MSL_EMAXITER, MSL_EDOM
from .utility import MinResult


comptime _GOLDEN: Float64 = 0.3819660112501051  # (3 - sqrt(5)) / 2
comptime _SQRT_DBL_EPSILON: Float64 = 1.4901161193847656e-8


# ===----------------------------------------------------------------------=== #
# Brent minimization
# ===----------------------------------------------------------------------=== #


def min_brent[fn_: def(Float64) capturing -> Float64](
    a: Float64,
    x: Float64,
    b: Float64,
    epsabs: Float64 = 1e-10,
    epsrel: Float64 = 1e-4,
    max_iter: Int = 100,
) -> MinResult:
    """Find a minimum in (a, b) with initial guess x via Brent's method.

    Uses parabolic interpolation when reliable, golden section otherwise.
    Typically requires O(log(1/eps)) iterations.

    Parameters:
        fn_: Scalar function to minimize.

    Args:
        a: Left bracket (f(x) < f(a) required).
        x: Initial best guess for the minimum.
        b: Right bracket (f(x) < f(b) required).
        epsabs: Absolute tolerance on bracket width.
        epsrel: Relative tolerance on bracket width.
        max_iter: Maximum iterations.

    Returns:
        MinResult with x (minimum location), fun, nit, nfev, success, errno.
    """
    var x_lo = a
    var x_hi = b
    var z = x
    var f_z = fn_(z)
    var nfev: Int = 1

    var w = z
    var v = z
    var f_w = f_z
    var f_v = f_z

    var d: Float64 = 0.0
    var e: Float64 = 0.0

    for i in range(max_iter):
        var tol = _SQRT_DBL_EPSILON * abs(z)
        var tol1 = epsabs + epsrel * abs(z)
        var tol2 = 2.0 * tol1
        var m = 0.5 * (x_lo + x_hi)

        if abs(z - m) <= tol2 - 0.5 * (x_hi - x_lo):
            return MinResult(x=z, fun=f_z, nit=i + 1, nfev=nfev, success=True)

        var use_parabola = False
        if abs(e) > tol:
            # Fit parabola through v, w, z
            var r = (z - w) * (f_z - f_v)
            var q = (z - v) * (f_z - f_w)
            var p = (z - v) * q - (z - w) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            else:
                q = -q
            r = e
            e = d
            if abs(p) < abs(0.5 * q * r) and p > q * (x_lo - z) and p < q * (x_hi - z):
                d = p / q
                use_parabola = True
                if (z + d - x_lo) < tol2 or (x_hi - z - d) < tol2:
                    d = tol if z < m else -tol

        if not use_parabola:
            # Golden section step into larger sub-interval
            e = (x_hi if z < m else x_lo) - z
            d = _GOLDEN * e

        var u = z + (d if abs(d) >= tol else (tol if d > 0.0 else -tol))
        var f_u = fn_(u)
        nfev += 1

        if f_u <= f_z:
            if u < z:
                x_hi = z
            else:
                x_lo = z
            v = w; f_v = f_w
            w = z; f_w = f_z
            z = u; f_z = f_u
        else:
            if u < z:
                x_lo = u
            else:
                x_hi = u
            if f_u <= f_w or w == z:
                v = w; f_v = f_w
                w = u; f_w = f_u
            elif f_u <= f_v or v == z or v == w:
                v = u; f_v = f_u

    return MinResult(x=z, fun=f_z, nit=max_iter, nfev=nfev, success=False, errno=MSL_EMAXITER)


# ===----------------------------------------------------------------------=== #
# Golden section search
# ===----------------------------------------------------------------------=== #


def min_golden[fn_: def(Float64) capturing -> Float64](
    a: Float64,
    x: Float64,
    b: Float64,
    epsabs: Float64 = 1e-10,
    epsrel: Float64 = 1e-4,
    max_iter: Int = 200,
) -> MinResult:
    """Find a minimum in (a, b) via golden section search.

    Parameters:
        fn_: Scalar function to minimize.

    Args:
        a: Left bracket.
        x: Initial best guess (unused internally, kept for API symmetry).
        b: Right bracket.
        epsabs: Absolute tolerance on bracket width.
        epsrel: Relative tolerance on bracket width.
        max_iter: Maximum iterations.

    Returns:
        MinResult with x (minimum location), fun, nit, nfev, success, errno.
    """
    var x_lo = a
    var x_hi = b
    comptime invphi: Float64 = 0.6180339887498948  # (sqrt(5) - 1) / 2

    var x_c = x_lo + invphi * (x_hi - x_lo)
    var x_d = x_hi - invphi * (x_hi - x_lo)
    var f_c = fn_(x_c)
    var f_d = fn_(x_d)
    var nfev: Int = 2

    for i in range(max_iter):
        var width = x_hi - x_lo
        var tol = epsabs + epsrel * abs(0.5 * (x_lo + x_hi))

        if width <= tol:
            var xm = 0.5 * (x_lo + x_hi)
            return MinResult(x=xm, fun=min(f_c, f_d), nit=i + 1, nfev=nfev, success=True)

        if f_c < f_d:
            x_hi = x_d
            x_d = x_c; f_d = f_c
            x_c = x_lo + invphi * (x_hi - x_lo)
            f_c = fn_(x_c)
            nfev += 1
        else:
            x_lo = x_c
            x_c = x_d; f_c = f_d
            x_d = x_hi - invphi * (x_hi - x_lo)
            f_d = fn_(x_d)
            nfev += 1

    var xm = 0.5 * (x_lo + x_hi)
    return MinResult(x=xm, fun=min(f_c, f_d), nit=max_iter, nfev=nfev, success=False, errno=MSL_EMAXITER)
