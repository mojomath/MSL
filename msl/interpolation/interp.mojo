# SPDX-License-Identifier: GPL-3.0-or-later

# ===----------------------------------------------------------------------=== #
# MSL (Mojo Scientific Library)
#
# Derived from GNU Scientific Library (GSL) and NuEvolve-mojo
#
# Original GSL authors:
# Copyright (C) 1996-2007 Gerard Jungman, Brian Gough
#
# Original NuEvolve authors:
# Shivasankar K.A.
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
1-D and 2-D interpolation.

Provides four scalar interpolation types:

  LinearInterp    - piecewise linear, O(1) eval
  CubicSpline     - natural cubic spline, O(n) init, O(log n) eval
  AkimaSpline     - Akima cubic spline (local, handles outliers), O(n) init
  BiCubicInterp   - 2-D bicubic built on Akima derivatives

All 1-D types expose: eval, deriv, deriv2, integral.
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.math import abs
from std.memory import (
    Pointer,
    unsafe_memset_zero,
)
from std.memory.alloc import unsafe_alloc

# ===----------------------------------------------------------------------=== #
# MSL
# ===----------------------------------------------------------------------=== #
from msl.core.errno import MSL_EDOM


# ===----------------------------------------------------------------------=== #
# Result types
# ===----------------------------------------------------------------------=== #


struct InterpResult(Copyable, Movable):
    """Result of an interpolation evaluation."""

    var val: Float64
    var errno: Int

    def __init__(out self, val: Float64 = 0.0, errno: Int = 0):
        self.val = val
        self.errno = errno

    def __init__(out self, *, copy: Self):
        self.val = copy.val
        self.errno = copy.errno

    def __init__(out self, *, deinit move: Self):
        self.val = move.val
        self.errno = move.errno


# ===----------------------------------------------------------------------=== #
# Internal helpers
# ===----------------------------------------------------------------------=== #


def _bsearch[
    mut: Bool, origin: Origin[mut=mut], //
](xa: Pointer[Float64, origin], n: Int, x: Float64) -> Int:
    """Binary search: return index i such that xa[i] <= x < xa[i+1]."""
    var lo: Int = 0
    var hi: Int = n - 1
    while hi - lo > 1:
        var mid = (lo + hi) // 2
        if xa[unsafe_offset=mid] <= x:
            lo = mid
        else:
            hi = mid
    return lo


def _alloc_f64(n: Int) -> Pointer[Float64, MutUntrackedOrigin]:
    var p = unsafe_alloc[Float64](n)
    unsafe_memset_zero(p, n)
    return p


# ===----------------------------------------------------------------------=== #
# Linear interpolation
# ===----------------------------------------------------------------------=== #


struct LinearInterp[mut: Bool, origin: Origin[mut=mut], //](Movable):
    """Piecewise linear interpolation."""

    var _xa: Pointer[Float64, Self.origin]
    var _ya: Pointer[Float64, Self.origin]
    var _n: Int

    def __init__(
        out self,
        xa: Pointer[Float64, Self.origin],
        ya: Pointer[Float64, Self.origin],
        n: Int,
    ):
        self._xa = xa
        self._ya = ya
        self._n = n

    def __init__(out self, *, deinit move: Self):
        self._xa = move._xa
        self._ya = move._ya
        self._n = move._n

    def eval(self, x: Float64) -> InterpResult:
        """Evaluate interpolant at x."""
        if x < self._xa[unsafe_offset=0] or x > self._xa[unsafe_offset=self._n - 1]:
            return InterpResult(errno=MSL_EDOM)
        var i = _bsearch(self._xa, self._n, x)
        var x_lo = self._xa[unsafe_offset=i]
        var x_hi = self._xa[unsafe_offset=i + 1]
        var y_lo = self._ya[unsafe_offset=i]
        var y_hi = self._ya[unsafe_offset=i + 1]
        var dx = x_hi - x_lo
        return InterpResult((y_lo * (x_hi - x) + y_hi * (x - x_lo)) / dx)

    def deriv(self, x: Float64) -> InterpResult:
        """First derivative (piecewise constant)."""
        if x < self._xa[unsafe_offset=0] or x > self._xa[unsafe_offset=self._n - 1]:
            return InterpResult(errno=MSL_EDOM)
        var i = _bsearch(self._xa, self._n, x)
        var dx = self._xa[unsafe_offset=i + 1] - self._xa[unsafe_offset=i]
        return InterpResult((self._ya[unsafe_offset=i + 1] - self._ya[unsafe_offset=i]) / dx)

    def deriv2(self, x: Float64) -> InterpResult:
        """Second derivative (zero everywhere for linear)."""
        if x < self._xa[unsafe_offset=0] or x > self._xa[unsafe_offset=self._n - 1]:
            return InterpResult(errno=MSL_EDOM)
        return InterpResult(0.0)

    def integral(self, a: Float64, b: Float64) -> InterpResult:
        """Definite integral from a to b via trapezoid rule."""
        if a < self._xa[unsafe_offset=0] or b > self._xa[unsafe_offset=self._n - 1]:
            return InterpResult(errno=MSL_EDOM)
        var sum: Float64 = 0.0
        var i_a = _bsearch(self._xa, self._n, a)
        var i_b = _bsearch(self._xa, self._n, b)

        var r_a = self.eval(a)
        var r_mid_a = self.eval(self._xa[unsafe_offset=i_a + 1])
        sum += 0.5 * (r_a.val + r_mid_a.val) * (self._xa[unsafe_offset=i_a + 1] - a)

        for i in range(i_a + 1, i_b):
            var h = self._xa[unsafe_offset=i + 1] - self._xa[unsafe_offset=i]
            sum += 0.5 * (self._ya[unsafe_offset=i] + self._ya[unsafe_offset=i + 1]) * h

        if i_b > i_a:
            var r_b = self.eval(b)
            sum += 0.5 * (self._ya[unsafe_offset=i_b] + r_b.val) * (b - self._xa[unsafe_offset=i_b])
        return InterpResult(sum)


# ===----------------------------------------------------------------------=== #
# Natural cubic spline
# ===----------------------------------------------------------------------=== #


struct CubicSpline[mut: Bool, origin: Origin[mut=mut], //](Movable):
    """Natural cubic spline interpolation.

    Second derivatives at endpoints are zero (natural boundary conditions).
    Coefficients b, c, d stored per interval; c[i] = second derivative / 2.
    """

    comptime MutExt = MutUntrackedOrigin

    var _xa: Pointer[Float64, Self.origin]
    var _ya: Pointer[Float64, Self.origin]
    var _n: Int
    var _b: Pointer[Float64, Self.MutExt]
    var _c: Pointer[Float64, Self.MutExt]
    var _d: Pointer[Float64, Self.MutExt]

    def __init__(
        out self,
        xa: Pointer[Float64, Self.origin],
        ya: Pointer[Float64, Self.origin],
        n: Int,
    ):
        self._xa = xa
        self._ya = ya
        self._n = n
        self._b = _alloc_f64(n)
        self._c = _alloc_f64(n)
        self._d = _alloc_f64(n)
        self._build()

    def __init__(out self, *, deinit move: Self):
        self._xa = move._xa
        self._ya = move._ya
        self._n = move._n
        self._b = move._b
        self._c = move._c
        self._d = move._d

    def __deinit__(deinit self):
        self._b.unsafe_free()
        self._c.unsafe_free()
        self._d.unsafe_free()

    def _build(mut self):
        """Solve tridiagonal system for second derivatives."""
        var n = self._n
        var g = _alloc_f64(n)
        var diag = _alloc_f64(n)
        var offdiag = _alloc_f64(n)

        # Build RHS and diag/offdiag
        for i in range(1, n - 1):
            var h_i = self._xa[unsafe_offset=i] - self._xa[unsafe_offset=i - 1]
            var h_ip1 = self._xa[unsafe_offset=i + 1] - self._xa[unsafe_offset=i]
            var dy_i = (self._ya[unsafe_offset=i] - self._ya[unsafe_offset=i - 1]) / h_i
            var dy_ip1 = (self._ya[unsafe_offset=i + 1] - self._ya[unsafe_offset=i]) / h_ip1
            offdiag[unsafe_offset=i] = h_ip1
            diag[unsafe_offset=i] = 2.0 * (h_i + h_ip1)
            g[unsafe_offset=i] = 3.0 * (dy_ip1 - dy_i)

        self._c[unsafe_offset=0] = 0.0
        self._c[unsafe_offset=n - 1] = 0.0

        # Thomas algorithm
        for i in range(1, n - 1):
            var h_i = self._xa[unsafe_offset=i] - self._xa[unsafe_offset=i - 1]
            if i > 1:
                var factor = h_i / diag[unsafe_offset=i - 1]
                diag[unsafe_offset=i] -= factor * offdiag[unsafe_offset=i - 1]
                g[unsafe_offset=i] -= factor * g[unsafe_offset=i - 1]

        # Back substitution
        if n > 2:
            self._c[unsafe_offset=n - 2] = g[unsafe_offset=n - 2] / diag[unsafe_offset=n - 2]
        for i in range(n - 3, 0, -1):
            var h_ip1 = self._xa[unsafe_offset=i + 1] - self._xa[unsafe_offset=i]
            self._c[unsafe_offset=i] = (g[unsafe_offset=i] - h_ip1 * self._c[unsafe_offset=i + 1]) / diag[unsafe_offset=i]

        for i in range(n - 1):
            var h = self._xa[unsafe_offset=i + 1] - self._xa[unsafe_offset=i]
            var dy = (self._ya[unsafe_offset=i + 1] - self._ya[unsafe_offset=i]) / h
            self._b[unsafe_offset=i] = dy - h * (self._c[unsafe_offset=i + 1] + 2.0 * self._c[unsafe_offset=i]) / 3.0
            self._d[unsafe_offset=i] = (self._c[unsafe_offset=i + 1] - self._c[unsafe_offset=i]) / (3.0 * h)

        g.unsafe_free()
        diag.unsafe_free()
        offdiag.unsafe_free()

    def eval(self, x: Float64) -> InterpResult:
        """Evaluate spline at x."""
        if x < self._xa[unsafe_offset=0] or x > self._xa[unsafe_offset=self._n - 1]:
            return InterpResult(errno=MSL_EDOM)
        var i = _bsearch(self._xa, self._n, x)
        var dx = x - self._xa[unsafe_offset=i]
        var val = self._ya[unsafe_offset=i] + dx * (
            self._b[unsafe_offset=i] + dx * (self._c[unsafe_offset=i] + dx * self._d[unsafe_offset=i])
        )
        return InterpResult(val)

    def deriv(self, x: Float64) -> InterpResult:
        """First derivative at x."""
        if x < self._xa[unsafe_offset=0] or x > self._xa[unsafe_offset=self._n - 1]:
            return InterpResult(errno=MSL_EDOM)
        var i = _bsearch(self._xa, self._n, x)
        var dx = x - self._xa[unsafe_offset=i]
        var val = self._b[unsafe_offset=i] + dx * (2.0 * self._c[unsafe_offset=i] + 3.0 * self._d[unsafe_offset=i] * dx)
        return InterpResult(val)

    def deriv2(self, x: Float64) -> InterpResult:
        """Second derivative at x."""
        if x < self._xa[unsafe_offset=0] or x > self._xa[unsafe_offset=self._n - 1]:
            return InterpResult(errno=MSL_EDOM)
        var i = _bsearch(self._xa, self._n, x)
        var dx = x - self._xa[unsafe_offset=i]
        var val = 2.0 * self._c[unsafe_offset=i] + 6.0 * self._d[unsafe_offset=i] * dx
        return InterpResult(val)

    def integral(self, a: Float64, b: Float64) -> InterpResult:
        """Definite integral from a to b."""
        if a < self._xa[unsafe_offset=0] or b > self._xa[unsafe_offset=self._n - 1]:
            return InterpResult(errno=MSL_EDOM)
        var sum: Float64 = 0.0
        var i_a = _bsearch(self._xa, self._n, a)
        var i_b = _bsearch(self._xa, self._n, b)

        for i in range(i_a, i_b + 1):
            var x_lo = a if i == i_a else self._xa[unsafe_offset=i]
            var x_hi = b if i == i_b else self._xa[unsafe_offset=i + 1]
            var da = x_lo - self._xa[unsafe_offset=i]
            var db = x_hi - self._xa[unsafe_offset=i]
            sum += (
                self._ya[unsafe_offset=i] * (db - da)
                + self._b[unsafe_offset=i] * (db * db - da * da) * 0.5
                + self._c[unsafe_offset=i] * (db * db * db - da * da * da) / 3.0
                + self._d[unsafe_offset=i] * (db * db * db * db - da * da * da * da) * 0.25
            )
        return InterpResult(sum)


# ===----------------------------------------------------------------------=== #
# Akima spline
# ===----------------------------------------------------------------------=== #


struct AkimaSpline[mut: Bool, origin: Origin[mut=mut], //](Movable):
    """Akima cubic spline interpolation.

    Uses weighted slope averages to compute Hermite cubic coefficients.
    More robust than cubic spline for data with outliers.
    """

    comptime MutExt = MutUntrackedOrigin

    var _xa: Pointer[Float64, Self.origin]
    var _ya: Pointer[Float64, Self.origin]
    var _n: Int
    var _b: Pointer[Float64, Self.MutExt]
    var _c: Pointer[Float64, Self.MutExt]
    var _d: Pointer[Float64, Self.MutExt]

    def __init__(
        out self,
        xa: Pointer[Float64, Self.origin],
        ya: Pointer[Float64, Self.origin],
        n: Int,
    ):
        self._xa = xa
        self._ya = ya
        self._n = n
        self._b = _alloc_f64(n)
        self._c = _alloc_f64(n)
        self._d = _alloc_f64(n)
        self._build()

    def __init__(out self, *, deinit move: Self):
        self._xa = move._xa
        self._ya = move._ya
        self._n = move._n
        self._b = move._b
        self._c = move._c
        self._d = move._d

    def __deinit__(deinit self):
        self._b.unsafe_free()
        self._c.unsafe_free()
        self._d.unsafe_free()

    def _build(mut self):
        """Compute Akima coefficients."""
        var n = self._n
        # m[i] = finite difference slope for interval i, with offset +2 for phantom points
        # total = n-1 intervals + 4 phantom = n+3 slopes, stored at offset 2
        var m_size = n + 3
        var m = _alloc_f64(m_size)

        # Interior slopes: m[i+2] = (y[i+1]-y[i])/(x[i+1]-x[i])
        for i in range(n - 1):
            m[unsafe_offset=i + 2] = (self._ya[unsafe_offset=i + 1] - self._ya[unsafe_offset=i]) / (
                self._xa[unsafe_offset=i + 1] - self._xa[unsafe_offset=i]
            )

        m[unsafe_offset=1] = 2.0 * m[unsafe_offset=2] - m[unsafe_offset=3]
        m[unsafe_offset=0] = 2.0 * m[unsafe_offset=1] - m[unsafe_offset=2]
        m[unsafe_offset=n + 1] = 2.0 * m[unsafe_offset=n] - m[unsafe_offset=n - 1]
        m[unsafe_offset=n + 2] = 2.0 * m[unsafe_offset=n + 1] - m[unsafe_offset=n]

        for i in range(n - 1):
            var h = self._xa[unsafe_offset=i + 1] - self._xa[unsafe_offset=i]
            var m0 = m[unsafe_offset=i]
            var m1 = m[unsafe_offset=i + 1]
            var m2 = m[unsafe_offset=i + 2]
            var m3 = m[unsafe_offset=i + 3]
            var m4 = m[unsafe_offset=i + 4] if i + 4 < m_size else m[unsafe_offset=m_size - 1]

            # Left slope at node i
            var ne_l = abs(m3 - m2) + abs(m1 - m0)
            var t_l: Float64
            if ne_l == 0.0:
                t_l = 0.5 * (m1 + m2)
            else:
                t_l = (abs(m3 - m2) * m1 + abs(m1 - m0) * m2) / ne_l

            # Left slope at node i+1
            var ne_r = abs(m4 - m3) + abs(m2 - m1)
            var t_r: Float64
            if ne_r == 0.0:
                t_r = 0.5 * (m2 + m3)
            else:
                t_r = (abs(m4 - m3) * m2 + abs(m2 - m1) * m3) / ne_r

            self._b[unsafe_offset=i] = t_l
            self._c[unsafe_offset=i] = (3.0 * m2 - 2.0 * t_l - t_r) / h
            self._d[unsafe_offset=i] = (t_l + t_r - 2.0 * m2) / (h * h)

        m.unsafe_free()

    def eval(self, x: Float64) -> InterpResult:
        """Evaluate spline at x."""
        if x < self._xa[unsafe_offset=0] or x > self._xa[unsafe_offset=self._n - 1]:
            return InterpResult(errno=MSL_EDOM)
        var i = _bsearch(self._xa, self._n, x)
        var dx = x - self._xa[unsafe_offset=i]
        var val = self._ya[unsafe_offset=i] + dx * (
            self._b[unsafe_offset=i] + dx * (self._c[unsafe_offset=i] + dx * self._d[unsafe_offset=i])
        )
        return InterpResult(val)

    def deriv(self, x: Float64) -> InterpResult:
        """First derivative at x."""
        if x < self._xa[unsafe_offset=0] or x > self._xa[unsafe_offset=self._n - 1]:
            return InterpResult(errno=MSL_EDOM)
        var i = _bsearch(self._xa, self._n, x)
        var dx = x - self._xa[unsafe_offset=i]
        var val = self._b[unsafe_offset=i] + dx * (2.0 * self._c[unsafe_offset=i] + 3.0 * self._d[unsafe_offset=i] * dx)
        return InterpResult(val)

    def deriv2(self, x: Float64) -> InterpResult:
        """Second derivative at x."""
        if x < self._xa[unsafe_offset=0] or x > self._xa[unsafe_offset=self._n - 1]:
            return InterpResult(errno=MSL_EDOM)
        var i = _bsearch(self._xa, self._n, x)
        var dx = x - self._xa[unsafe_offset=i]
        var val = 2.0 * self._c[unsafe_offset=i] + 6.0 * self._d[unsafe_offset=i] * dx
        return InterpResult(val)

    def integral(self, a: Float64, b: Float64) -> InterpResult:
        """Definite integral from a to b."""
        if a < self._xa[unsafe_offset=0] or b > self._xa[unsafe_offset=self._n - 1]:
            return InterpResult(errno=MSL_EDOM)
        var sum: Float64 = 0.0
        var i_a = _bsearch(self._xa, self._n, a)
        var i_b = _bsearch(self._xa, self._n, b)

        for i in range(i_a, i_b + 1):
            var x_lo = a if i == i_a else self._xa[unsafe_offset=i]
            var x_hi = b if i == i_b else self._xa[unsafe_offset=i + 1]
            var da = x_lo - self._xa[unsafe_offset=i]
            var db = x_hi - self._xa[unsafe_offset=i]
            sum += (
                self._ya[unsafe_offset=i] * (db - da)
                + self._b[unsafe_offset=i] * (db * db - da * da) * 0.5
                + self._c[unsafe_offset=i] * (db * db * db - da * da * da) / 3.0
                + self._d[unsafe_offset=i] * (db * db * db * db - da * da * da * da) * 0.25
            )
        return InterpResult(sum)
