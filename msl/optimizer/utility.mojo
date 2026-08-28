# SPDX-License-Identifier: GPL-3.0-or-later

# ===----------------------------------------------------------------------=== #
# MSL (Mojo Scientific Library)
#
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
"""Result types and helper routines for scalar optimization."""

from std.math import abs

from msl.core.const import MSL_SQRT_DBL_EPSILON
from msl.core.errno import MSL_SUCCESS
from msl.core.errno import (
    MSL_FAILURE,
    MSL_CONTINUE,
    MSL_EBADTOL,
    MSL_EINVAL,
)


struct RootResult(Copyable, Movable):
    """Result of a scalar root-finding operation."""

    var root: Float64
    """Root estimate."""
    var nit: Int
    """Number of iterations performed."""
    var nfev: Int
    """Number of function evaluations."""
    var success: Bool
    """True if converged within tolerance."""
    var errno: Int
    """Error code (0 = success)."""

    def __init__(
        out self,
        root: Float64 = 0.0,
        nit: Int = 0,
        nfev: Int = 0,
        success: Bool = False,
        errno: Int = MSL_SUCCESS,
    ):
        self.root = root
        self.nit = nit
        self.nfev = nfev
        self.success = success
        self.errno = errno

    def __init__(out self, *, copy: Self):
        self.root = copy.root
        self.nit = copy.nit
        self.nfev = copy.nfev
        self.success = copy.success
        self.errno = copy.errno

    def __init__(out self, *, deinit move: Self):
        self.root = move.root
        self.nit = move.nit
        self.nfev = move.nfev
        self.success = move.success
        self.errno = move.errno


struct MinResult(Copyable, Movable):
    """Result of a scalar minimization operation."""

    var x: Float64
    """Location of the minimum."""
    var fun: Float64
    """Function value at the minimum."""
    var nit: Int
    """Number of iterations performed."""
    var nfev: Int
    """Number of function evaluations."""
    var success: Bool
    """True if converged within tolerance."""
    var errno: Int
    """Error code (0 = success)."""

    def __init__(
        out self,
        x: Float64 = 0.0,
        fun: Float64 = 0.0,
        nit: Int = 0,
        nfev: Int = 0,
        success: Bool = False,
        errno: Int = MSL_SUCCESS,
    ):
        self.x = x
        self.fun = fun
        self.nit = nit
        self.nfev = nfev
        self.success = success
        self.errno = errno

    def __init__(out self, *, copy: Self):
        self.x = copy.x
        self.fun = copy.fun
        self.nit = copy.nit
        self.nfev = copy.nfev
        self.success = copy.success
        self.errno = copy.errno

    def __init__(out self, *, deinit move: Self):
        self.x = move.x
        self.fun = move.fun
        self.nit = move.nit
        self.nfev = move.nfev
        self.success = move.success
        self.errno = move.errno


struct MinBracketResult(Copyable, Movable):
    """Result of minimization bracketing."""

    var x_minimum: Float64
    var f_minimum: Float64
    var x_lower: Float64
    var f_lower: Float64
    var x_upper: Float64
    var f_upper: Float64
    var nfev: Int
    var success: Bool
    var errno: Int

    def __init__(
        out self,
        x_minimum: Float64 = 0.0,
        f_minimum: Float64 = 0.0,
        x_lower: Float64 = 0.0,
        f_lower: Float64 = 0.0,
        x_upper: Float64 = 0.0,
        f_upper: Float64 = 0.0,
        nfev: Int = 0,
        success: Bool = False,
        errno: Int = MSL_SUCCESS,
    ):
        self.x_minimum = x_minimum
        self.f_minimum = f_minimum
        self.x_lower = x_lower
        self.f_lower = f_lower
        self.x_upper = x_upper
        self.f_upper = f_upper
        self.nfev = nfev
        self.success = success
        self.errno = errno

    def __init__(out self, *, copy: Self):
        self.x_minimum = copy.x_minimum
        self.f_minimum = copy.f_minimum
        self.x_lower = copy.x_lower
        self.f_lower = copy.f_lower
        self.x_upper = copy.x_upper
        self.f_upper = copy.f_upper
        self.nfev = copy.nfev
        self.success = copy.success
        self.errno = copy.errno

    def __init__(out self, *, deinit move: Self):
        self.x_minimum = move.x_minimum
        self.f_minimum = move.f_minimum
        self.x_lower = move.x_lower
        self.f_lower = move.f_lower
        self.x_upper = move.x_upper
        self.f_upper = move.f_upper
        self.nfev = move.nfev
        self.success = move.success
        self.errno = move.errno


def root_test_interval(
    x_lower: Float64, x_upper: Float64, epsabs: Float64, epsrel: Float64
) -> Int:
    """Convergence test for bracketing root solvers."""
    if epsrel < 0.0 or epsabs < 0.0:
        return MSL_EBADTOL
    if x_lower > x_upper:
        return MSL_EINVAL

    var abs_lower = abs(x_lower)
    var abs_upper = abs(x_upper)
    var min_abs: Float64 = 0.0

    if (x_lower > 0.0 and x_upper > 0.0) or (x_lower < 0.0 and x_upper < 0.0):
        min_abs = abs_lower if abs_lower < abs_upper else abs_upper

    var tolerance = epsabs + epsrel * min_abs
    if abs(x_upper - x_lower) < tolerance:
        return MSL_SUCCESS
    return MSL_CONTINUE


def root_test_delta(
    x1: Float64, x0: Float64, epsabs: Float64, epsrel: Float64
) -> Int:
    """Convergence test for step-size changes in root solvers."""
    if epsrel < 0.0 or epsabs < 0.0:
        return MSL_EBADTOL

    var tolerance = epsabs + epsrel * abs(x1)
    if abs(x1 - x0) < tolerance or x1 == x0:
        return MSL_SUCCESS
    return MSL_CONTINUE


def root_test_residual(f: Float64, epsabs: Float64) -> Int:
    """Convergence test for root residual values."""
    if epsabs < 0.0:
        return MSL_EBADTOL

    if abs(f) < epsabs:
        return MSL_SUCCESS
    return MSL_CONTINUE


def min_test_interval(
    x_lower: Float64, x_upper: Float64, epsabs: Float64, epsrel: Float64
) -> Int:
    """Convergence test for minimization bracket intervals."""
    if epsrel < 0.0 or epsabs < 0.0:
        return MSL_EBADTOL
    if x_lower > x_upper:
        return MSL_EINVAL

    var abs_lower = abs(x_lower)
    var abs_upper = abs(x_upper)
    var min_abs: Float64 = 0.0

    if (x_lower > 0.0 and x_upper > 0.0) or (x_lower < 0.0 and x_upper < 0.0):
        min_abs = abs_lower if abs_lower < abs_upper else abs_upper

    var tolerance = epsabs + epsrel * min_abs
    if abs(x_upper - x_lower) < tolerance:
        return MSL_SUCCESS
    return MSL_CONTINUE


def min_find_bracket[
    fn_: def(Float64) capturing -> Float64
](
    x_lower0: Float64,
    x_upper0: Float64,
    eval_max: Int = 100,
) -> MinBracketResult:
    """Find an initial bracket [x_lower, x_upper] containing a local minimum."""
    comptime golden: Float64 = 0.3819660

    var x_left = x_lower0
    var x_right = x_upper0
    var f_left = fn_(x_left)
    var f_right = fn_(x_right)
    var nfev: Int = 2

    var x_center: Float64
    var f_center: Float64

    if f_right >= f_left:
        x_center = (x_right - x_left) * golden + x_left
        f_center = fn_(x_center)
        nfev += 1
    else:
        x_center = x_right
        f_center = f_right
        x_right = (x_center - x_left) / golden + x_left
        f_right = fn_(x_right)
        nfev += 1

    while (
        nfev < eval_max
        and (x_right - x_left)
        > MSL_SQRT_DBL_EPSILON * ((x_right + x_left) * 0.5)
        + MSL_SQRT_DBL_EPSILON
    ):
        if f_center < f_left:
            if f_center < f_right:
                return MinBracketResult(
                    x_minimum=x_center,
                    f_minimum=f_center,
                    x_lower=x_left,
                    f_lower=f_left,
                    x_upper=x_right,
                    f_upper=f_right,
                    nfev=nfev,
                    success=True,
                )
            elif f_center > f_right:
                x_left = x_center
                f_left = f_center
                x_center = x_right
                f_center = f_right
                x_right = (x_center - x_left) / golden + x_left
                f_right = fn_(x_right)
                nfev += 1
            else:
                x_right = x_center
                f_right = f_center
                x_center = (x_right - x_left) * golden + x_left
                f_center = fn_(x_center)
                nfev += 1
        else:
            x_right = x_center
            f_right = f_center
            x_center = (x_right - x_left) * golden + x_left
            f_center = fn_(x_center)
            nfev += 1

    return MinBracketResult(
        x_minimum=x_center,
        f_minimum=f_center,
        x_lower=x_left,
        f_lower=f_left,
        x_upper=x_right,
        f_upper=f_right,
        nfev=nfev,
        success=False,
        errno=MSL_FAILURE,
    )
