# SPDX-License-Identifier: GPL-3.0-or-later

# ===----------------------------------------------------------------------=== #
# MSL (Mojo Scientific Library)
#
# Derived from GNU Scientific Library (GSL)
# Original file: deriv/deriv.c
#
# Original authors:
# Copyright (C) 2004, 2007 Brian Gough
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
Numerical differentiation.

Three routines are:

  deriv_central  — 5-point symmetric rule + 3-point embedded for error,
                   with optional adaptive step refinement.
  deriv_forward  — 4-point forward rule at x+h/4..x+h, with adaptive
                   step refinement.
  deriv_backward — forward rule evaluated at -h (same stencil, mirrored).

All three separate truncation error from rounding error and optionally
refine h to minimise the total.
"""

from std.math import abs, pow

from msl.core.const import MSL_DBL_EPSILON


# ===----------------------------------------------------------------------=== #
# Result type
# ===----------------------------------------------------------------------=== #


struct DerivResult(Copyable, Movable):
    """Result of a numerical differentiation."""

    var val: Float64
    """Derivative estimate."""
    var err: Float64
    """Absolute error estimate (truncation + rounding)."""

    def __init__(out self, val: Float64, err: Float64):
        self.val = val
        self.err = err

    def __init__(out self):
        self.val = 0.0
        self.err = 0.0

    def __init__(out self, *, copy: Self):
        self.val = copy.val
        self.err = copy.err

    def __init__(out self, *, deinit take: Self):
        self.val = take.val
        self.err = take.err


# ===----------------------------------------------------------------------=== #
# Internal: central_deriv (5-point / 3-point)
# ===----------------------------------------------------------------------=== #


def _central_deriv[fn_: def(Float64) capturing -> Float64](
    x: Float64, h: Float64
) -> Tuple[Float64, Float64, Float64]:
    """5-point central rule with embedded 3-point for error decomposition.

    Returns (result, abserr_round, abserr_trunc).
    """
    var fm1 = fn_(x - h)
    var fp1 = fn_(x + h)
    var fmh = fn_(x - h * 0.5)
    var fph = fn_(x + h * 0.5)

    var r3 = 0.5 * (fp1 - fm1)
    var r5 = (4.0 / 3.0) * (fph - fmh) - (1.0 / 3.0) * r3

    var e3 = (abs(fp1) + abs(fm1)) * MSL_DBL_EPSILON
    var e5 = 2.0 * (abs(fph) + abs(fmh)) * MSL_DBL_EPSILON + e3

    # Finite precision in x+h: O(eps * x / h)
    var dy = max(abs(r3 / h), abs(r5 / h)) * (abs(x) / h) * MSL_DBL_EPSILON

    var result = r5 / h
    var abserr_trunc = abs((r5 - r3) / h)
    var abserr_round = abs(e5 / h) + dy
    return (result, abserr_round, abserr_trunc)


# ===----------------------------------------------------------------------=== #
# Internal: forward_deriv (4-point at h/4, h/2, 3h/4, h)
# ===----------------------------------------------------------------------=== #


def _forward_deriv[fn_: def(Float64) capturing -> Float64](
    x: Float64, h: Float64
) -> Tuple[Float64, Float64, Float64]:
    """4-point forward rule with embedded 2-point for error decomposition.

    Returns (result, abserr_round, abserr_trunc).
    """
    var f1 = fn_(x + h * 0.25)
    var f2 = fn_(x + h * 0.5)
    var f3 = fn_(x + h * 0.75)
    var f4 = fn_(x + h)

    var r2 = 2.0 * (f4 - f2)
    var r4 = (22.0 / 3.0) * (f4 - f3) - (62.0 / 3.0) * (f3 - f2) + (52.0 / 3.0) * (f2 - f1)

    var e4 = 2.0 * 20.67 * (abs(f4) + abs(f3) + abs(f2) + abs(f1)) * MSL_DBL_EPSILON
    var dy = max(abs(r2 / h), abs(r4 / h)) * abs(x / h) * MSL_DBL_EPSILON

    var result = r4 / h
    var abserr_trunc = abs((r4 - r2) / h)
    var abserr_round = abs(e4 / h) + dy
    return (result, abserr_round, abserr_trunc)


# ===----------------------------------------------------------------------=== #
# Public API
# ===----------------------------------------------------------------------=== #


def deriv_central[fn_: def(Float64) capturing -> Float64](
    x: Float64, h: Float64 = 1e-4
) -> DerivResult:
    """First derivative via 5-point central differences.

    Uses the 5-point rule (x±h, x±h/2) with an embedded 3-point rule
    (x±h) for error estimation. If rounding error is smaller than
    truncation error, refines h once to minimise the total error.

    Parameters:
        fn_: Function to differentiate, must have `capturing` effect.

    Args:
        x: Point at which to evaluate the derivative.
        h: Initial step size (default 1e-4).

    Returns:
        DerivResult with val (derivative) and err (truncation + rounding).
    """
    var r0: Float64
    var round_err: Float64
    var trunc_err: Float64
    r0, round_err, trunc_err = _central_deriv[fn_](x, h)
    var error = round_err + trunc_err

    if round_err < trunc_err and round_err > 0.0 and trunc_err > 0.0:
        var h_opt = h * pow(round_err / (2.0 * trunc_err), 1.0 / 3.0)
        var r_opt: Float64
        var round_opt: Float64
        var trunc_opt: Float64
        r_opt, round_opt, trunc_opt = _central_deriv[fn_](x, h_opt)
        var error_opt = round_opt + trunc_opt
        if error_opt < error and abs(r_opt - r0) < 4.0 * error:
            r0 = r_opt
            error = error_opt

    return DerivResult(r0, error)


def deriv_forward[fn_: def(Float64) capturing -> Float64](
    x: Float64, h: Float64 = 1e-4
) -> DerivResult:
    """First derivative via 4-point forward differences.

    Uses evaluations at x+h/4, x+h/2, x+3h/4, x+h. Suitable for
    left-boundary points where f(x-h) is unavailable. Refines h
    once if rounding error is dominant.

    Parameters:
        fn_: Function to differentiate, must have `capturing` effect.

    Args:
        x: Point at which to evaluate the derivative.
        h: Initial step size (default 1e-4, must be > 0).

    Returns:
        DerivResult with val (derivative) and err (truncation + rounding).
    """
    var r0: Float64
    var round_err: Float64
    var trunc_err: Float64
    r0, round_err, trunc_err = _forward_deriv[fn_](x, h)
    var error = round_err + trunc_err

    if round_err < trunc_err and round_err > 0.0 and trunc_err > 0.0:
        var h_opt = h * pow(round_err / trunc_err, 0.5)
        var r_opt: Float64
        var round_opt: Float64
        var trunc_opt: Float64
        r_opt, round_opt, trunc_opt = _forward_deriv[fn_](x, h_opt)
        var error_opt = round_opt + trunc_opt
        if error_opt < error and abs(r_opt - r0) < 4.0 * error:
            r0 = r_opt
            error = error_opt

    return DerivResult(r0, error)


def deriv_backward[fn_: def(Float64) capturing -> Float64](
    x: Float64, h: Float64 = 1e-4
) -> DerivResult:
    """First derivative via backward differences.

    Equivalent to deriv_forward evaluated at -h — the same 4-point
    stencil mirrored. Suitable for right-boundary points.

    Parameters:
        fn_: Function to differentiate, must have `capturing` effect.

    Args:
        x: Point at which to evaluate the derivative.
        h: Initial step size (default 1e-4, must be > 0).

    Returns:
        DerivResult with val (derivative) and err (truncation + rounding).
    """
    return deriv_forward[fn_](x, -h)
