# SPDX-License-Identifier: GPL-3.0-or-later

# ===----------------------------------------------------------------------=== #
# MSL (Mojo Scientific Library)
#
# Derived from GNU Scientific Library (GSL)
# Original files:
#   min/fsolver.c
#   min/gsl_min.h
#
# Original authors:
# Copyright (C) 1996-2009 Brian Gough
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
MinFSolver - bracketing minimizer (brent or golden).
"""

from std.math import abs, sqrt

from msl.core.const import MSL_DBL_EPSILON
from msl.core.errno import MSL_SUCCESS, MSL_EMAXITER, MSL_EDOM
from .utility import MinResult, min_test_interval


comptime _GOLDEN_MIN: Float64 = 0.3819660112501051
comptime _SQRT_DBL_EPSILON: Float64 = 1.4901161193847656e-8


struct MinFSolver[
    fn_: def(Float64) capturing -> Float64,
    method: StringLiteral = "brent",
](Movable):
    """Stateful bracketing minimizer.

    Parameters:
        fn_: Function to minimise.
        method: "brent" or "golden" (default "brent").

    Example:
        ```mojo
        from msl.core import MSL_SUCCESS
        from msl.optimizer import MinFSolver, min_test_interval

        def func(x: Float64) capturing -> Float64: return (x - 1.5) * (x - 1.5)

        var s = MinFSolver[func]()
        s.set(1.0, 0.0, 3.0)
        for _ in range(100):
            s.iterate()
            if min_test_interval(s.x_lower(), s.x_upper(), 1e-10, 0.0) == MSL_SUCCESS:
                break
        print(s.x_minimum())
        ```

    Call `set(x_minimum, x_lower, x_upper)` to initialise, then call
    `iterate()` in a loop and check convergence with `min_test_interval`.
    """

    var _x_minimum: Float64
    var _x_lower: Float64
    var _x_upper: Float64
    var _f_minimum: Float64
    var _f_lower: Float64
    var _f_upper: Float64

    # brent state
    var _v: Float64
    var _w: Float64
    var _f_v: Float64
    var _f_w: Float64
    var _d: Float64
    var _e: Float64

    var _nit: Int

    def __init__(out self):
        self._x_minimum = 0.0
        self._x_lower = 0.0
        self._x_upper = 0.0
        self._f_minimum = 0.0
        self._f_lower = 0.0
        self._f_upper = 0.0
        self._v = 0.0
        self._w = 0.0
        self._f_v = 0.0
        self._f_w = 0.0
        self._d = 0.0
        self._e = 0.0
        self._nit = 0

    def set(
        mut self,
        x_minimum: Float64,
        x_lower: Float64,
        x_upper: Float64,
    ) -> Int:
        """Initialise with bracket (x_lower, x_minimum, x_upper).

        Returns MSL_SUCCESS or MSL_EDOM if the bracket is invalid.
        """
        var f_lower = Self.fn_(x_lower)
        var f_minimum = Self.fn_(x_minimum)
        var f_upper = Self.fn_(x_upper)
        return self.set_with_values(
            x_minimum, f_minimum, x_lower, f_lower, x_upper, f_upper
        )

    def set_with_values(
        mut self,
        x_minimum: Float64,
        f_minimum: Float64,
        x_lower: Float64,
        f_lower: Float64,
        x_upper: Float64,
        f_upper: Float64,
    ) -> Int:
        """Initialise with precomputed function values."""
        if f_minimum >= f_lower or f_minimum >= f_upper:
            return MSL_EDOM
        self._x_minimum = x_minimum
        self._x_lower = x_lower
        self._x_upper = x_upper
        self._f_minimum = f_minimum
        self._f_lower = f_lower
        self._f_upper = f_upper
        self._nit = 0

        comptime if Self.method == "brent":
            self._v = x_minimum
            self._w = x_minimum
            self._f_v = f_minimum
            self._f_w = f_minimum
            self._d = 0.0
            self._e = 0.0

        return MSL_SUCCESS

    def iterate(mut self) -> Int:
        """Perform one iteration. Returns MSL_SUCCESS."""
        self._nit += 1

        comptime if Self.method == "golden":
            return self._iterate_golden()
        else:
            return self._iterate_brent()

    def _iterate_golden(mut self) -> Int:
        var x_m = self._x_minimum
        var x_l = self._x_lower
        var x_u = self._x_upper
        var x_new: Float64
        if x_u - x_m > x_m - x_l:
            x_new = x_m + _GOLDEN_MIN * (x_u - x_m)
        else:
            x_new = x_m - _GOLDEN_MIN * (x_m - x_l)
        var f_new = Self.fn_(x_new)
        if f_new < self._f_minimum:
            if x_new < x_m:
                self._x_upper = x_m
                self._f_upper = self._f_minimum
            else:
                self._x_lower = x_m
                self._f_lower = self._f_minimum
            self._x_minimum = x_new
            self._f_minimum = f_new
        else:
            if x_new < x_m:
                self._x_lower = x_new
                self._f_lower = f_new
            else:
                self._x_upper = x_new
                self._f_upper = f_new
        return MSL_SUCCESS

    def _iterate_brent(mut self) -> Int:
        var x_m = self._x_minimum
        var x_l = self._x_lower
        var x_u = self._x_upper
        var f_m = self._f_minimum
        var v = self._v
        var w = self._w
        var f_v = self._f_v
        var f_w = self._f_w
        var d = self._d
        var e = self._e

        var tol = _SQRT_DBL_EPSILON * abs(x_m)
        var midpoint = 0.5 * (x_l + x_u)
        var x_new: Float64
        var f_new: Float64

        if abs(e) > tol:
            var r = (x_m - w) * (f_m - f_v)
            var q2 = (x_m - v) * (f_m - f_w)
            var p = (x_m - v) * q2 - (x_m - w) * r
            q2 = 2.0 * (q2 - r)
            if q2 > 0.0:
                p = -p
            else:
                q2 = -q2
            r = e
            e = d
            if (
                abs(p) < abs(0.5 * q2 * r)
                and p > q2 * (x_l - x_m)
                and p < q2 * (x_u - x_m)
            ):
                d = p / q2
                x_new = x_m + d
                if (x_new - x_l) < 2.0 * tol or (x_u - x_new) < 2.0 * tol:
                    d = tol if midpoint > x_m else -tol
            else:
                e = (x_l - x_m) if x_m >= midpoint else (x_u - x_m)
                d = _GOLDEN_MIN * e
        else:
            e = (x_l - x_m) if x_m >= midpoint else (x_u - x_m)
            d = _GOLDEN_MIN * e

        x_new = x_m + (d if abs(d) >= tol else (tol if d > 0.0 else -tol))
        f_new = Self.fn_(x_new)

        if f_new <= f_m:
            if x_new < x_m:
                self._x_upper = x_m
                self._f_upper = f_m
            else:
                self._x_lower = x_m
                self._f_lower = f_m
            v = w
            f_v = f_w
            w = x_m
            f_w = f_m
            self._x_minimum = x_new
            self._f_minimum = f_new
        else:
            if x_new < x_m:
                self._x_lower = x_new
                self._f_lower = f_new
            else:
                self._x_upper = x_new
                self._f_upper = f_new
            if f_new <= f_w or w == x_m:
                v = w
                f_v = f_w
                w = x_new
                f_w = f_new
            elif f_new <= f_v or v == x_m or v == w:
                v = x_new
                f_v = f_new

        self._v = v
        self._w = w
        self._f_v = f_v
        self._f_w = f_w
        self._d = d
        self._e = e
        return MSL_SUCCESS

    def x_minimum(self) -> Float64:
        """Current minimum location estimate."""
        return self._x_minimum

    def x_lower(self) -> Float64:
        """Current lower bracket bound."""
        return self._x_lower

    def x_upper(self) -> Float64:
        """Current upper bracket bound."""
        return self._x_upper

    def f_minimum(self) -> Float64:
        """Function value at current minimum estimate."""
        return self._f_minimum

    def f_lower(self) -> Float64:
        """Function value at lower bracket bound."""
        return self._f_lower

    def f_upper(self) -> Float64:
        """Function value at upper bracket bound."""
        return self._f_upper

    def name(self) -> String:
        """Algorithm name."""
        return Self.method

    def solve(
        mut self,
        x_minimum: Float64,
        x_lower: Float64,
        x_upper: Float64,
        epsabs: Float64 = 1e-10,
        epsrel: Float64 = 1e-4,
        max_iter: Int = 100,
    ) -> MinResult:
        """Convenience: set bracket and iterate to convergence."""
        var status = self.set(x_minimum, x_lower, x_upper)
        if status != MSL_SUCCESS:
            return MinResult(errno=status)
        for i in range(max_iter):
            _ = self.iterate()
            if (
                min_test_interval(self._x_lower, self._x_upper, epsabs, epsrel)
                == MSL_SUCCESS
            ):
                return MinResult(
                    x=self._x_minimum,
                    fun=self._f_minimum,
                    nit=i + 1,
                    nfev=self._nit,
                    success=True,
                )
        return MinResult(
            x=self._x_minimum,
            fun=self._f_minimum,
            nit=max_iter,
            nfev=self._nit,
            success=False,
            errno=MSL_EMAXITER,
        )
