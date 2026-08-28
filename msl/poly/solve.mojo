# SPDX-License-Identifier: GPL-3.0-or-later

# ===----------------------------------------------------------------------=== #
# MSL (Mojo Scientific Library)
#
# Derived from GNU Scientific Library (GSL)
# Original files:
#   poly/solve_quadratic.c
#   poly/solve_cubic.c
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
Real-root solvers for quadratic and cubic polynomials (P1).

Complex-root variants (`gsl_poly_complex_solve_quadratic/cubic`) are
deferred until MSL has a Complex type.
"""

from std.math import abs, acos, cos, pow, sqrt

comptime _M_PI: Float64 = 3.14159265358979323846


struct QuadraticRoots(Copyable, Movable):
    """Result of solving a x^2 + b x + c = 0 for real roots."""

    var x0: Float64
    """First (smaller or equal) real root, if any."""
    var x1: Float64
    """Second (larger or equal) real root, if any."""
    var nroots: Int
    """Number of real roots found (0, 1, or 2)."""

    def __init__(out self, x0: Float64 = 0.0, x1: Float64 = 0.0, nroots: Int = 0):
        self.x0 = x0
        self.x1 = x1
        self.nroots = nroots

    def __init__(out self, *, copy: Self):
        self.x0 = copy.x0
        self.x1 = copy.x1
        self.nroots = copy.nroots

    def __init__(out self, *, deinit move: Self):
        self.x0 = move.x0
        self.x1 = move.x1
        self.nroots = move.nroots


struct CubicRoots(Copyable, Movable):
    """Result of solving x^3 + a x^2 + b x + c = 0 for real roots."""

    var x0: Float64
    """First (smallest) real root."""
    var x1: Float64
    """Second real root, if any."""
    var x2: Float64
    """Third (largest) real root, if any."""
    var nroots: Int
    """Number of real roots found (1 or 3)."""

    def __init__(
        out self,
        x0: Float64 = 0.0,
        x1: Float64 = 0.0,
        x2: Float64 = 0.0,
        nroots: Int = 0,
    ):
        self.x0 = x0
        self.x1 = x1
        self.x2 = x2
        self.nroots = nroots

    def __init__(out self, *, copy: Self):
        self.x0 = copy.x0
        self.x1 = copy.x1
        self.x2 = copy.x2
        self.nroots = copy.nroots

    def __init__(out self, *, deinit move: Self):
        self.x0 = move.x0
        self.x1 = move.x1
        self.x2 = move.x2
        self.nroots = move.nroots


def poly_solve_quadratic(a: Float64, b: Float64, c: Float64) -> QuadraticRoots:
    """Find the real roots of a x^2 + b x + c = 0.

    Roots are returned ordered (x0 <= x1). nroots is 0, 1, or 2.
    """
    if a == 0.0:
        if b == 0.0:
            return QuadraticRoots(nroots=0)
        return QuadraticRoots(x0=-c / b, x1=-c / b, nroots=1)

    var disc = b * b - 4.0 * a * c

    if disc > 0.0:
        if b == 0.0:
            var r = sqrt(-c / a)
            return QuadraticRoots(x0=-r, x1=r, nroots=2)
        var sgnb: Float64 = 1.0 if b > 0.0 else -1.0
        var temp = -0.5 * (b + sgnb * sqrt(disc))
        var r1 = temp / a
        var r2 = c / temp
        if r1 < r2:
            return QuadraticRoots(x0=r1, x1=r2, nroots=2)
        return QuadraticRoots(x0=r2, x1=r1, nroots=2)
    elif disc == 0.0:
        var r = -0.5 * b / a
        return QuadraticRoots(x0=r, x1=r, nroots=2)
    else:
        return QuadraticRoots(nroots=0)


def poly_solve_cubic(a: Float64, b: Float64, c: Float64) -> CubicRoots:
    """Find the real roots of x^3 + a x^2 + b x + c = 0.

    Roots are returned ordered. nroots is 1 or 3.
    """
    var q = a * a - 3.0 * b
    var r = 2.0 * a * a * a - 9.0 * a * b + 27.0 * c

    var Q = q / 9.0
    var R = r / 54.0

    var Q3 = Q * Q * Q
    var R2 = R * R

    var CR2 = 729.0 * r * r
    var CQ3 = 2916.0 * q * q * q

    if R == 0.0 and Q == 0.0:
        var x = -a / 3.0
        return CubicRoots(x0=x, x1=x, x2=x, nroots=3)
    elif CR2 == CQ3:
        var sqrtQ = sqrt(Q)
        if R > 0.0:
            return CubicRoots(
                x0=-2.0 * sqrtQ - a / 3.0,
                x1=sqrtQ - a / 3.0,
                x2=sqrtQ - a / 3.0,
                nroots=3,
            )
        return CubicRoots(
            x0=-sqrtQ - a / 3.0,
            x1=-sqrtQ - a / 3.0,
            x2=2.0 * sqrtQ - a / 3.0,
            nroots=3,
        )
    elif R2 < Q3:
        var sgnR: Float64 = 1.0 if R >= 0.0 else -1.0
        var ratio = sgnR * sqrt(R2 / Q3)
        var theta = acos(ratio)
        var norm = -2.0 * sqrt(Q)
        var x0 = norm * cos(theta / 3.0) - a / 3.0
        var x1 = norm * cos((theta + 2.0 * _M_PI) / 3.0) - a / 3.0
        var x2 = norm * cos((theta - 2.0 * _M_PI) / 3.0) - a / 3.0

        if x0 > x1:
            x0, x1 = x1, x0
        if x1 > x2:
            x1, x2 = x2, x1
            if x0 > x1:
                x0, x1 = x1, x0

        return CubicRoots(x0=x0, x1=x1, x2=x2, nroots=3)
    else:
        var sgnR: Float64 = 1.0 if R >= 0.0 else -1.0
        var A = -sgnR * pow(abs(R) + sqrt(R2 - Q3), 1.0 / 3.0)
        var B = Q / A
        return CubicRoots(x0=A + B - a / 3.0, nroots=1)
