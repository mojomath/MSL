# SPDX-License-Identifier: GPL-3.0-or-later

# ===----------------------------------------------------------------------=== #
# MSL (Mojo Scientific Library)
#
# Derived from GNU Scientific Library (GSL)
# Original files: ode-initval2/rk4.c, ode-initval2/rkf45.c,
#                 ode-initval2/driver.c, ode-initval2/cstd.c
#
# Original authors:
# Copyright (C) 1996-2007 Gerard Jungman, Brian Gough
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
ODE initial value problem solvers.

  ode_rk4    - Classic 4th-order Runge-Kutta, fixed step size.
  ode_rkf45  - Runge-Kutta-Fehlberg 4(5), adaptive step size.

Both solve dy/dt = f(t, y) for a system of `dim` equations.
The integrand signature is:

    def rhs(t: Float64, y: Pointer[Float64, MutExt],
            dydt: Pointer[Float64, MutExt]) capturing:
        dydt[0] = ...
        dydt[1] = ...
"""

from std.math import abs, pow, sqrt
from std.memory import Pointer, unsafe_memset_zero

from msl.core.const import MSL_DBL_EPSILON, MSL_DBL_MAX
from msl.core.errno import MSL_SUCCESS, MSL_EMAXITER, MSL_EINVAL


# ===----------------------------------------------------------------------=== #
# Result type
# ===----------------------------------------------------------------------=== #


struct OdeResult(Copyable, Movable):
    """Result of an ODE integration."""

    var t: Float64
    """Final time reached."""
    var nsteps: Int
    """Number of steps taken."""
    var nfev: Int
    """Number of right-hand-side evaluations."""
    var success: Bool
    """True if integration reached t_end within tolerance."""
    var errno: Int
    """Error code (0 = success)."""

    def __init__(
        out self,
        t: Float64 = 0.0,
        nsteps: Int = 0,
        nfev: Int = 0,
        success: Bool = False,
        errno: Int = MSL_SUCCESS,
    ):
        self.t = t
        self.nsteps = nsteps
        self.nfev = nfev
        self.success = success
        self.errno = errno

    def __init__(out self, *, copy: Self):
        self.t = copy.t
        self.nsteps = copy.nsteps
        self.nfev = copy.nfev
        self.success = copy.success
        self.errno = copy.errno

    def __init__(out self, *, deinit move: Self):
        self.t = move.t
        self.nsteps = move.nsteps
        self.nfev = move.nfev
        self.success = move.success
        self.errno = move.errno


# ===----------------------------------------------------------------------=== #
# Internal workspace helpers
# ===----------------------------------------------------------------------=== #


def _alloc(n: Int) -> Pointer[Float64, MutExt]:
    var p = unsafe_alloc[Float64](n)
    unsafe_memset_zero(p, n)
    return p


# ===----------------------------------------------------------------------=== #
# Classical RK4 - fixed step
# ===----------------------------------------------------------------------=== #


def ode_rk4[
    origin_y: MutOrigin,
    //,
    rhs: def[origin_fn_y: MutOrigin, origin_fn_dydt: MutOrigin, //](
        Float64,
        Pointer[Float64, origin_fn_y],
        Pointer[Float64, origin_fn_dydt],
    ) capturing,
](
    t0: Float64,
    t1: Float64,
    h: Float64,
    y: Pointer[Float64, origin_y],
    dim: Int,
    max_steps: Int = 100000,
) -> OdeResult:
    """Integrate dy/dt = f(t,y) from t0 to t1 using classical RK4.

    The y array is updated in-place with the solution at t1.

    Parameters:
        rhs: Right-hand-side function f(t, y, dydt) with `capturing` effect.

    Args:
        t0: Start time.
        t1: End time.
        h: Step size (must be > 0; last step is shortened to reach t1 exactly).
        y: Initial condition on input; solution at t1 on output (length dim).
        dim: Number of equations.
        max_steps: Safety cap on iterations.

    Returns:
        OdeResult with t (final time), nsteps, nfev, success, errno.
    """
    if h <= 0.0 or dim <= 0:
        return OdeResult(errno=MSL_EINVAL)

    var k1 = _alloc(dim)
    var k2 = _alloc(dim)
    var k3 = _alloc(dim)
    var k4 = _alloc(dim)
    var ytmp = _alloc(dim)

    var t = t0
    var nsteps: Int = 0
    var nfev: Int = 0

    while t < t1 and nsteps < max_steps:
        var step = min(h, t1 - t)

        # k1 = f(t, y)
        rhs(t, y, k1)
        nfev += 1

        # k2 = f(t + h/2, y + h/2 * k1)
        for i in range(dim):
            ytmp[i] = y[i] + 0.5 * step * k1[i]
        rhs(t + 0.5 * step, ytmp, k2)
        nfev += 1

        # k3 = f(t + h/2, y + h/2 * k2)
        for i in range(dim):
            ytmp[i] = y[i] + 0.5 * step * k2[i]
        rhs(t + 0.5 * step, ytmp, k3)
        nfev += 1

        # k4 = f(t + h, y + h * k3)
        for i in range(dim):
            ytmp[i] = y[i] + step * k3[i]
        rhs(t + step, ytmp, k4)
        nfev += 1

        # y_new = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        for i in range(dim):
            y[i] += (step / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])

        t += step
        nsteps += 1

    k1.unsafe_free()
    k2.unsafe_free()
    k3.unsafe_free()
    k4.unsafe_free()
    ytmp.unsafe_free()

    if nsteps >= max_steps and t < t1 - MSL_DBL_EPSILON * abs(t1):
        return OdeResult(
            t=t, nsteps=nsteps, nfev=nfev, success=False, errno=MSL_EMAXITER
        )

    return OdeResult(t=t, nsteps=nsteps, nfev=nfev, success=True)


# ===----------------------------------------------------------------------=== #
# RKF45 - adaptive step (Runge-Kutta-Fehlberg)
# ===----------------------------------------------------------------------=== #

# Butcher tableau for RKF45
comptime _AH: InlineArray[Float64, 5] = [
    1.0 / 4.0,
    3.0 / 8.0,
    12.0 / 13.0,
    1.0,
    1.0 / 2.0,
]
comptime _B3: InlineArray[Float64, 2] = [3.0 / 32.0, 9.0 / 32.0]
comptime _B4: InlineArray[Float64, 3] = [
    1932.0 / 2197.0,
    -7200.0 / 2197.0,
    7296.0 / 2197.0,
]
comptime _B5: InlineArray[Float64, 4] = [
    8341.0 / 4104.0,
    -32832.0 / 4104.0,
    29440.0 / 4104.0,
    -845.0 / 4104.0,
]
comptime _B6: InlineArray[Float64, 5] = [
    -6080.0 / 20520.0,
    41040.0 / 20520.0,
    -28352.0 / 20520.0,
    9295.0 / 20520.0,
    -5643.0 / 20520.0,
]
# 5th-order output weights (indices 0,2 are zero)
comptime _C: InlineArray[Float64, 6] = [
    902880.0 / 7618050.0,
    0.0,
    3953664.0 / 7618050.0,
    3855735.0 / 7618050.0,
    -1371249.0 / 7618050.0,
    277020.0 / 7618050.0,
]
# Error coefficients (difference 5th - 4th order)
comptime _EC: InlineArray[Float64, 7] = [
    0.0,
    1.0 / 360.0,
    0.0,
    -128.0 / 4275.0,
    -2197.0 / 75240.0,
    1.0 / 50.0,
    2.0 / 55.0,
]


def ode_rkf45[
    origin_y: MutOrigin,
    //,
    rhs: def[origin_fn_y: MutOrigin, origin_fn_dydt: MutOrigin, //](
        Float64,
        Pointer[Float64, origin_fn_y],
        Pointer[Float64, origin_fn_dydt],
    ) capturing,
](
    t0: Float64,
    t1: Float64,
    h0: Float64,
    y: Pointer[Float64, origin_y],
    dim: Int,
    epsabs: Float64 = 1e-6,
    epsrel: Float64 = 1e-6,
    hmin: Float64 = 0.0,
    hmax: Float64 = 0.0,
    max_steps: Int = 100000,
) -> OdeResult:
    """Integrate dy/dt = f(t,y) from t0 to t1 using RKF45 with adaptive step size.

    The y array is updated in-place with the solution at t1.

    Parameters:
        rhs: Right-hand-side function f(t, y, dydt) with `capturing` effect.

    Args:
        t0: Start time.
        t1: End time.
        h0: Initial step size hint.
        y: Initial condition on input; solution at t1 on output (length dim).
        dim: Number of equations.
        epsabs: Absolute error tolerance per component.
        epsrel: Relative error tolerance per component.
        hmin: Minimum allowed step size (0 = no minimum).
        hmax: Maximum allowed step size (0 = no maximum).
        max_steps: Safety cap on iterations.

    Returns:
        OdeResult with t (final time), nsteps, nfev, success, errno.
    """
    if h0 <= 0.0 or dim <= 0:
        return OdeResult(errno=MSL_EINVAL)

    var k1 = _alloc(dim)
    var k2 = _alloc(dim)
    var k3 = _alloc(dim)
    var k4 = _alloc(dim)
    var k5 = _alloc(dim)
    var k6 = _alloc(dim)
    var ytmp = _alloc(dim)
    var yerr = _alloc(dim)

    var h_max = (t1 - t0) if hmax <= 0.0 else hmax
    var h_min = 0.0 if hmin <= 0.0 else hmin

    var t = t0
    var h = min(h0, h_max)
    var nsteps: Int = 0
    var nfev: Int = 0

    while t < t1 and nsteps < max_steps:
        h = min(h, t1 - t)
        if h <= 0.0:
            break

        # k1
        rhs(t, y, k1)
        nfev += 1

        # k2
        for i in range(dim):
            ytmp[i] = y[i] + h * _AH[0] * k1[i]
        rhs(t + _AH[0] * h, ytmp, k2)
        nfev += 1

        # k3
        for i in range(dim):
            ytmp[i] = y[i] + h * (_B3[0] * k1[i] + _B3[1] * k2[i])
        rhs(t + _AH[1] * h, ytmp, k3)
        nfev += 1

        # k4
        for i in range(dim):
            ytmp[i] = y[i] + h * (
                _B4[0] * k1[i] + _B4[1] * k2[i] + _B4[2] * k3[i]
            )
        rhs(t + _AH[2] * h, ytmp, k4)
        nfev += 1

        # k5
        for i in range(dim):
            ytmp[i] = y[i] + h * (
                _B5[0] * k1[i]
                + _B5[1] * k2[i]
                + _B5[2] * k3[i]
                + _B5[3] * k4[i]
            )
        rhs(t + _AH[3] * h, ytmp, k5)
        nfev += 1

        # k6
        for i in range(dim):
            ytmp[i] = y[i] + h * (
                _B6[0] * k1[i]
                + _B6[1] * k2[i]
                + _B6[2] * k3[i]
                + _B6[3] * k4[i]
                + _B6[4] * k5[i]
            )
        rhs(t + _AH[4] * h, ytmp, k6)
        nfev += 1

        # Error estimate
        for i in range(dim):
            yerr[i] = h * (
                _EC[1] * k1[i]
                + _EC[3] * k3[i]
                + _EC[4] * k4[i]
                + _EC[5] * k5[i]
                + _EC[6] * k6[i]
            )

        # Step control: find max(|yerr[i]| / D0[i])
        var rmax: Float64 = MSL_DBL_EPSILON
        for i in range(dim):
            var D0 = epsrel * abs(y[i]) + epsabs
            var r = abs(yerr[i]) / D0
            if r > rmax:
                rmax = r

        if rmax <= 1.0:
            # Accept step - update y with 5th-order estimate
            for i in range(dim):
                y[i] += h * (
                    _C[0] * k1[i]
                    + _C[2] * k3[i]
                    + _C[3] * k4[i]
                    + _C[4] * k5[i]
                    + _C[5] * k6[i]
                )
            t += h
            nsteps += 1

        # Adjust step size (GSL safety factor S=0.9, order=5)
        var h_new = 0.9 * h * pow(1.0 / rmax, 0.2)
        h_new = min(h_new, 5.0 * h)  # cap growth
        h_new = max(h_new, 0.2 * h)  # cap shrinkage
        h_new = min(h_new, h_max)
        if h_min > 0.0:
            h_new = max(h_new, h_min)
        h = h_new

    k1.unsafe_free()
    k2.unsafe_free()
    k3.unsafe_free()
    k4.unsafe_free()
    k5.unsafe_free()
    k6.unsafe_free()
    ytmp.unsafe_free()
    yerr.unsafe_free()

    if nsteps >= max_steps and t < t1 - MSL_DBL_EPSILON * abs(t1):
        return OdeResult(
            t=t, nsteps=nsteps, nfev=nfev, success=False, errno=MSL_EMAXITER
        )

    return OdeResult(t=t, nsteps=nsteps, nfev=nfev, success=True)
