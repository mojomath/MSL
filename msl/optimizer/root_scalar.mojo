# SPDX-License-Identifier: GPL-3.0-or-later

# ===----------------------------------------------------------------------=== #
# MSL (Mojo Scientific Library)
#
# Derived from GNU Scientific Library (GSL)
# Original files: roots/bisection.c, roots/brent.c, roots/newton.c
#
# Original authors:
# Copyright (C) 1996-2007 Brian Gough, Reid Priedhorsky
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
Scalar root-finding algorithms.

  root_bisect  - Bisection (bracketing, robust, linear convergence)
  root_brent   - Brent's method (bracketing + interpolation, superlinear)
  root_newton  - Newton-Raphson (needs derivative, quadratic convergence)
  root_secant  - Secant method (derivative-free, superlinear)
"""

from std.math import abs

from msl.core.const import MSL_DBL_EPSILON
from msl.core.errno import MSL_SUCCESS, MSL_EMAXITER, MSL_EDOM
from .utility import RootResult


# ===----------------------------------------------------------------------=== #
# Bisection
# ===----------------------------------------------------------------------=== #


def root_bisect[fn_: def(Float64) capturing -> Float64](
    a: Float64,
    b: Float64,
    epsabs: Float64 = 1e-10,
    epsrel: Float64 = 1e-10,
    max_iter: Int = 100,
) -> RootResult:
    """Find a root in [a, b] via bisection.

    Requires f(a)*f(b) < 0. Guaranteed linear convergence (halves interval
    each step). Use when robustness matters more than speed.

    Parameters:
        fn_: Scalar function to find root of.

    Args:
        a: Left bracket (f(a) and f(b) must have opposite signs).
        b: Right bracket.
        epsabs: Absolute tolerance on half-bracket width.
        epsrel: Relative tolerance on half-bracket width.
        max_iter: Maximum iterations.

    Returns:
        RootResult with root, nit, nfev, success, errno.
    """
    var x_lo = a
    var x_hi = b
    var f_lo = fn_(x_lo)
    var f_hi = fn_(x_hi)
    var nfev: Int = 2

    if f_lo * f_hi > 0.0:
        return RootResult(errno=MSL_EDOM)

    for i in range(max_iter):
        var x_mid = 0.5 * (x_lo + x_hi)
        var f_mid = fn_(x_mid)
        nfev += 1

        if f_lo * f_mid <= 0.0:
            x_hi = x_mid
        else:
            x_lo = x_mid
            f_lo = f_mid

        var root = 0.5 * (x_lo + x_hi)
        var half_width = 0.5 * abs(x_hi - x_lo)
        var tol = epsabs + epsrel * abs(root)

        if half_width <= tol or f_mid == 0.0:
            return RootResult(root=root, nit=i + 1, nfev=nfev, success=True)

    return RootResult(
        root=0.5 * (x_lo + x_hi), nit=max_iter, nfev=nfev,
        success=False, errno=MSL_EMAXITER,
    )


# ===----------------------------------------------------------------------=== #
# Brent's method
# ===----------------------------------------------------------------------=== #


def root_brent[fn_: def(Float64) capturing -> Float64](
    a: Float64,
    b: Float64,
    epsabs: Float64 = 1e-10,
    epsrel: Float64 = 1e-10,
    max_iter: Int = 100,
) -> RootResult:
    """Find a root in [a, b] via Brent's method.

    Combines bisection, secant, and inverse quadratic interpolation.
    Superlinear convergence in practice with bisection fallback guarantee.

    Parameters:
        fn_: Scalar function to find root of.

    Args:
        a: Left bracket (f(a) and f(b) must have opposite signs).
        b: Right bracket.
        epsabs: Absolute tolerance.
        epsrel: Relative tolerance.
        max_iter: Maximum iterations.

    Returns:
        RootResult with root, nit, nfev, success, errno.
    """
    var x_a = a
    var x_b = b
    var fa = fn_(x_a)
    var fb = fn_(x_b)
    var nfev: Int = 2

    if fa * fb > 0.0:
        return RootResult(errno=MSL_EDOM)

    var x_c = x_a
    var fc = fa
    var d: Float64 = x_b - x_a
    var e: Float64 = d

    for i in range(max_iter):
        if fb * fc > 0.0:
            x_c = x_a; fc = fa
            d = x_b - x_a; e = d

        if abs(fc) < abs(fb):
            x_a = x_b; fa = fb
            x_b = x_c; fb = fc
            x_c = x_a; fc = fa

        var tol = 0.5 * MSL_DBL_EPSILON * abs(x_b)
        var m = 0.5 * (x_c - x_b)

        if abs(m) <= tol or fb == 0.0:
            return RootResult(root=x_b, nit=i + 1, nfev=nfev, success=True)

        if abs(e) >= tol and abs(fa) > abs(fb):
            var s = fb / fa
            var p: Float64
            var q: Float64
            if x_a == x_c:
                # Secant step
                p = 2.0 * m * s
                q = 1.0 - s
            else:
                # Inverse quadratic interpolation
                var r = fb / fc
                q = fa / fc
                p = s * (2.0 * m * q * (q - r) - (x_b - x_a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)

            if p > 0.0:
                q = -q
            else:
                p = -p

            if 2.0 * p < min(3.0 * m * q - abs(tol * q), abs(e * q)):
                e = d; d = p / q
            else:
                d = m; e = m
        else:
            d = m; e = m

        x_a = x_b; fa = fb

        if abs(d) > tol:
            x_b += d
        else:
            x_b += tol if m > 0.0 else -tol

        fb = fn_(x_b)
        nfev += 1

    return RootResult(
        root=x_b, nit=max_iter, nfev=nfev,
        success=False, errno=MSL_EMAXITER,
    )


# ===----------------------------------------------------------------------=== #
# Newton-Raphson
# ===----------------------------------------------------------------------=== #


def root_newton[
    fn_: def(Float64) capturing -> Float64,
    dfn_: def(Float64) capturing -> Float64,
](
    x0: Float64,
    epsabs: Float64 = 1e-10,
    epsrel: Float64 = 1e-10,
    max_iter: Int = 100,
) -> RootResult:
    """Find a root via Newton-Raphson iteration.

    Requires both f and f'. Quadratic convergence near the root.
    No bracketing - may diverge with poor initial guesses.

    Parameters:
        fn_: Scalar function to find root of.
        dfn_: Derivative of fn_.

    Args:
        x0: Initial guess.
        epsabs: Absolute tolerance on |f(x)|.
        epsrel: Relative tolerance on Newton step size.
        max_iter: Maximum iterations.

    Returns:
        RootResult with root, nit, nfev, success, errno.
    """
    var x = x0
    var nfev: Int = 0

    for i in range(max_iter):
        var fx = fn_(x)
        var dfx = dfn_(x)
        nfev += 2

        if dfx == 0.0:
            return RootResult(root=x, nit=i, nfev=nfev, success=False, errno=MSL_EDOM)

        var dx = fx / dfx
        x -= dx

        if abs(fx) <= epsabs or abs(dx) <= epsabs + epsrel * abs(x):
            return RootResult(root=x, nit=i + 1, nfev=nfev, success=True)

    return RootResult(root=x, nit=max_iter, nfev=nfev, success=False, errno=MSL_EMAXITER)


# ===----------------------------------------------------------------------=== #
# Secant method
# ===----------------------------------------------------------------------=== #


def root_secant[fn_: def(Float64) capturing -> Float64](
    x0: Float64,
    x1: Float64,
    epsabs: Float64 = 1e-10,
    epsrel: Float64 = 1e-10,
    max_iter: Int = 100,
) -> RootResult:
    """Find a root via the secant method.

    Derivative-free, order ~1.618 convergence. Requires two starting points.
    No bracketing guarantee - may diverge.

    Parameters:
        fn_: Scalar function to find root of.

    Args:
        x0: First initial point.
        x1: Second initial point (should be close to x0).
        epsabs: Absolute tolerance on |f(x)|.
        epsrel: Relative tolerance on step size.
        max_iter: Maximum iterations.

    Returns:
        RootResult with root, nit, nfev, success, errno.
    """
    var xa = x0
    var xb = x1
    var fa = fn_(xa)
    var fb = fn_(xb)
    var nfev: Int = 2

    for i in range(max_iter):
        var denom = fb - fa
        if denom == 0.0:
            return RootResult(root=xb, nit=i, nfev=nfev, success=False, errno=MSL_EDOM)

        var xc = xb - fb * (xb - xa) / denom
        var fc = fn_(xc)
        nfev += 1

        if abs(fc) <= epsabs or abs(xc - xb) <= epsabs + epsrel * abs(xc):
            return RootResult(root=xc, nit=i + 1, nfev=nfev, success=True)

        xa = xb; fa = fb
        xb = xc; fb = fc

    return RootResult(root=xb, nit=max_iter, nfev=nfev, success=False, errno=MSL_EMAXITER)
