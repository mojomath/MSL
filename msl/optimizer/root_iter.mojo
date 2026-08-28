# SPDX-License-Identifier: GPL-3.0-or-later

# ===----------------------------------------------------------------------=== #
# MSL (Mojo Scientific Library)
#
# Derived from GNU Scientific Library (GSL)
# Original files:
#   roots/fsolver.c
#   roots/fdfsolver.c
#   roots/gsl_roots.h
#
# Original authors:
# Copyright (C) 1996-2007 Reid Priedhorsky, Brian Gough
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
Stateful iterative root-finding solvers (R4).

  RootFSolver   - bracketing solver (bisect, brent, falsepos)
  RootFDFSolver - derivative-based solver (newton, secant, steffenson)

Usage example::
    ```mojo
    from msl.core import MSL_SUCCESS
    from msl.optimizer import RootFDFSolver, root_test_residual

    def f(x: Float64) -> Float64: return x * x - 2.0
    def df(x: Float64) -> Float64: return 2.0 * x

    var s = RootFDFSolver[f, df]()
    s.set(1.0)
    for _ in range(50):
        s.iterate()
        if root_test_residual(s.f(), 1e-10) == MSL_SUCCESS:
            break
    print(s.root())
    ```
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.math import abs

# ===----------------------------------------------------------------------=== #
# MSL
# ===----------------------------------------------------------------------=== #
from msl.core.const import MSL_DBL_EPSILON
from msl.core.errno import (
    MSL_EDOM,
    MSL_EMAXITER,
    MSL_EZERODIV,
    MSL_SUCCESS,
)
from msl.optimizer.utility import (
    root_test_interval,
    root_test_residual,
    RootResult,
)


# ===----------------------------------------------------------------------=== #
# RootFSolver  (bracketing: bisect / brent / falsepos)
# ===----------------------------------------------------------------------=== #


struct RootFSolver[
    fn_: def(Float64) capturing -> Float64,
    method: StringLiteral = "brent",
](Movable):
    """Stateful bracketing root solver.

    Parameters:
        fn_: Function to find root of.
        method: "bisect", "brent", or "falsepos" (default "brent").

    Call `set(x_lower, x_upper)` to initialise, then call `iterate()` in a
    loop and check convergence with `root_test_interval` or `root_test_residual`.
    """

    var _root: Float64
    var _x_lower: Float64
    var _x_upper: Float64
    var _f_lower: Float64
    var _f_upper: Float64

    # brent state
    var _x_c: Float64
    var _f_c: Float64
    var _d: Float64
    var _e: Float64

    # steffenson / falsepos step counter
    var _nit: Int

    def __init__(out self):
        self._root = 0.0
        self._x_lower = 0.0
        self._x_upper = 0.0
        self._f_lower = 0.0
        self._f_upper = 0.0
        self._x_c = 0.0
        self._f_c = 0.0
        self._d = 0.0
        self._e = 0.0
        self._nit = 0

    def set(mut self, x_lower: Float64, x_upper: Float64) -> Int:
        """Initialise the solver with a bracket [x_lower, x_upper].

        Returns MSL_SUCCESS or MSL_EDOM if f(x_lower)*f(x_upper) >= 0.
        """
        var fl = Self.fn_(x_lower)
        var fu = Self.fn_(x_upper)
        if fl * fu > 0.0:
            return MSL_EDOM

        self._x_lower = x_lower
        self._x_upper = x_upper
        self._f_lower = fl
        self._f_upper = fu
        self._root = x_lower if abs(fl) < abs(fu) else x_upper
        self._nit = 0

        comptime if Self.method == "brent":
            self._x_c = x_lower
            self._f_c = fl
            self._d = x_upper - x_lower
            self._e = self._d

        return MSL_SUCCESS

    def iterate(mut self) -> Int:
        """Perform one iteration. Returns MSL_SUCCESS or MSL_CONTINUE."""
        self._nit += 1

        comptime if Self.method == "bisect":
            return self._iterate_bisect()
        elif Self.method == "falsepos":
            return self._iterate_falsepos()
        else:
            return self._iterate_brent()

    def _iterate_bisect(mut self) -> Int:
        var x_mid = 0.5 * (self._x_lower + self._x_upper)
        var f_mid = Self.fn_(x_mid)
        if self._f_lower * f_mid <= 0.0:
            self._x_upper = x_mid
            self._f_upper = f_mid
        else:
            self._x_lower = x_mid
            self._f_lower = f_mid
        self._root = 0.5 * (self._x_lower + self._x_upper)
        return MSL_SUCCESS

    def _iterate_falsepos(mut self) -> Int:
        var denom = self._f_upper - self._f_lower
        if denom == 0.0:
            return MSL_EDOM
        var x_new = (
            self._x_upper
            - self._f_upper * (self._x_lower - self._x_upper) / denom
        )
        var f_new = Self.fn_(x_new)
        if self._f_lower * f_new <= 0.0:
            self._x_upper = x_new
            self._f_upper = f_new
        else:
            self._x_lower = x_new
            self._f_lower = f_new
        self._root = x_new
        return MSL_SUCCESS

    def _iterate_brent(mut self) -> Int:
        var x_a = self._x_lower
        var x_b = self._x_upper
        var fa = self._f_lower
        var fb = self._f_upper
        var x_c = self._x_c
        var fc = self._f_c
        var d = self._d
        var e = self._e

        if fb * fc > 0.0:
            x_c = x_a
            fc = fa
            d = x_b - x_a
            e = d

        if abs(fc) < abs(fb):
            x_a = x_b
            fa = fb
            x_b = x_c
            fb = fc
            x_c = x_a
            fc = fa

        var tol = 0.5 * MSL_DBL_EPSILON * abs(x_b)
        var m = 0.5 * (x_c - x_b)
        var x_new: Float64

        if abs(e) >= tol and abs(fa) > abs(fb):
            var s = fb / fa
            var p: Float64
            var q: Float64
            if x_a == x_c:
                p = 2.0 * m * s
                q = 1.0 - s
            else:
                var r = fb / fc
                q = fa / fc
                p = s * (2.0 * m * q * (q - r) - (x_b - x_a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0.0:
                q = -q
            else:
                p = -p
            if 2.0 * p < min(3.0 * m * q - abs(tol * q), abs(e * q)):
                e = d
                d = p / q
            else:
                d = m
                e = m
        else:
            d = m
            e = m

        x_new = x_b + (d if abs(d) > tol else (tol if m > 0.0 else -tol))
        var f_new = Self.fn_(x_new)

        if f_new * fb <= 0.0:
            self._x_lower = x_b
            self._f_lower = fb
            self._x_upper = x_new
            self._f_upper = f_new
        else:
            self._x_lower = x_new
            self._f_lower = f_new
            self._x_upper = x_b
            self._f_upper = fb

        self._x_c = x_c
        self._f_c = fc
        self._d = d
        self._e = e
        self._root = x_new
        return MSL_SUCCESS

    def root(self) -> Float64:
        """Current root estimate."""
        return self._root

    def x_lower(self) -> Float64:
        """Current lower bracket bound."""
        return self._x_lower

    def x_upper(self) -> Float64:
        """Current upper bracket bound."""
        return self._x_upper

    def f(self) -> Float64:
        """Function evaluated at current root estimate."""
        return Self.fn_(self._root)

    def name(self) -> String:
        """Algorithm name."""
        return Self.method

    def solve(
        mut self,
        x_lower: Float64,
        x_upper: Float64,
        epsabs: Float64 = 1e-10,
        epsrel: Float64 = 1e-10,
        max_iter: Int = 100,
    ) -> RootResult:
        """Convenience: set bracket and iterate to convergence."""
        var status = self.set(x_lower, x_upper)
        if status != MSL_SUCCESS:
            return RootResult(errno=status)
        for i in range(max_iter):
            _ = self.iterate()
            if (
                root_test_interval(self._x_lower, self._x_upper, epsabs, epsrel)
                == MSL_SUCCESS
            ):
                return RootResult(
                    root=self._root, nit=i + 1, nfev=self._nit, success=True
                )
        return RootResult(
            root=self._root,
            nit=max_iter,
            nfev=self._nit,
            success=False,
            errno=MSL_EMAXITER,
        )


# ===----------------------------------------------------------------------=== #
# RootFDFSolver  (derivative-based: newton / secant / steffenson)
# ===----------------------------------------------------------------------=== #


struct RootFDFSolver[
    fn_: def(Float64) capturing -> Float64,
    dfn_: def(Float64) capturing -> Float64,
    method: StringLiteral = "newton",
](Movable):
    """Stateful derivative-based root solver.

    Parameters:
        fn_: Function to find root of.
        dfn_: Derivative of fn_.
        method: "newton", "secant", or "steffenson" (default "newton").

    Call `set(x0)` to initialise, then call `iterate()` in a loop.
    """

    var _root: Float64
    var _f: Float64
    var _df: Float64

    # secant: previous point
    var _x_prev: Float64
    var _f_prev: Float64

    # steffenson: aitken state
    var _x_1: Float64
    var _count: Int

    var _nit: Int

    def __init__(out self):
        self._root = 0.0
        self._f = 0.0
        self._df = 0.0
        self._x_prev = 0.0
        self._f_prev = 0.0
        self._x_1 = 0.0
        self._count = 0
        self._nit = 0

    def set(mut self, x0: Float64):
        """Initialise the solver at starting point x0."""
        self._root = x0
        self._f = Self.fn_(x0)
        self._df = Self.dfn_(x0)
        self._x_prev = x0
        self._f_prev = self._f
        self._x_1 = x0
        self._count = 1
        self._nit = 0

    def iterate(mut self) -> Int:
        """Perform one iteration. Returns MSL_SUCCESS, MSL_EDOM, or MSL_EZERODIV.
        """
        self._nit += 1

        comptime if Self.method == "secant":
            return self._iterate_secant()
        elif Self.method == "steffenson":
            return self._iterate_steffenson()
        else:
            return self._iterate_newton()

    def _iterate_newton(mut self) -> Int:
        if self._df == 0.0:
            return MSL_EZERODIV
        var x_new = self._root - self._f / self._df
        self._x_prev = self._root
        self._root = x_new
        self._f = Self.fn_(x_new)
        self._df = Self.dfn_(x_new)
        return MSL_SUCCESS

    def _iterate_secant(mut self) -> Int:
        var denom = self._f - self._f_prev
        if denom == 0.0:
            return MSL_EZERODIV
        var x_new = self._root - self._f * (self._root - self._x_prev) / denom
        self._x_prev = self._root
        self._f_prev = self._f
        self._root = x_new
        self._f = Self.fn_(x_new)
        self._df = Self.dfn_(x_new)
        return MSL_SUCCESS

    def _iterate_steffenson(mut self) -> Int:
        if self._df == 0.0:
            return MSL_EZERODIV
        var x_new = self._root - self._f / self._df
        var f_new = Self.fn_(x_new)
        var df_new = Self.dfn_(x_new)

        var accel: Float64
        if self._count < 3:
            accel = x_new
            self._count += 1
        else:
            var u = self._root - self._x_1
            var v = x_new - 2.0 * self._root + self._x_1
            accel = x_new if v == 0.0 else (self._x_1 - u * u / v)

        self._x_1 = self._root
        self._root = accel
        self._f = f_new
        self._df = df_new
        return MSL_SUCCESS

    def root(self) -> Float64:
        """Current root estimate."""
        return self._root

    def f(self) -> Float64:
        """Function value at current root estimate."""
        return self._f

    def df(self) -> Float64:
        """Function derivative value at current root estimate."""
        return self._df

    def name(self) -> String:
        """Algorithm name."""
        return Self.method

    def solve(
        mut self,
        x0: Float64,
        epsabs: Float64 = 1e-10,
        epsrel: Float64 = 1e-10,
        max_iter: Int = 100,
    ) -> RootResult:
        """Convenience: set initial point and iterate to convergence."""
        self.set(x0)
        for i in range(max_iter):
            var status = self.iterate()
            if status != MSL_SUCCESS:
                return RootResult(
                    root=self._root,
                    nit=i,
                    nfev=self._nit,
                    success=False,
                    errno=status,
                )
            if root_test_residual(self._f, epsabs) == MSL_SUCCESS:
                return RootResult(
                    root=self._root, nit=i + 1, nfev=self._nit, success=True
                )
        return RootResult(
            root=self._root,
            nit=max_iter,
            nfev=self._nit,
            success=False,
            errno=MSL_EMAXITER,
        )
