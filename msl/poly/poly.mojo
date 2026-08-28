# SPDX-License-Identifier: GPL-3.0-or-later

# ===----------------------------------------------------------------------=== #
# MSL (Mojo Scientific Library)
#
# Derived from GNU Scientific Library (GSL)
# Original files:
#   poly/gsl_poly.h
#   poly/eval.c
#   poly/deriv.c
#   poly/dd.c
#
# Original authors:
# Copyright (C) 1996-2007 Brian Gough, Gerard Jungman
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
Real-valued polynomial evaluation and divided-difference interpolation (P1).

Complex-root solving (`gsl_poly_complex_solve` and the complex-argument
evaluators) is deferred until MSL has a Complex type and an eigenvalue
solver (see roadmap phase L1).
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.memory import Pointer


def poly_eval[
    mut: Bool, origin: Origin[mut=mut], //
](c: Pointer[Float64, origin], len: Int, x: Float64) -> Float64:
    """Evaluate c[0] + c[1] x + ... + c[len-1] x^(len-1) via Horner's method."""
    var ans = c[len - 1]
    for i in range(len - 1, 0, -1):
        ans = c[i - 1] + x * ans
    return ans


def poly_eval_derivs[
    mut: Bool,
    origin: Origin[mut=mut],
    res_origin: MutOrigin,
    //,
](
    c: Pointer[Float64, origin],
    lenc: Int,
    x: Float64,
    res: Pointer[Float64, res_origin],
    lenres: Int,
):
    """Evaluate the polynomial and its derivatives at x.

    res[k] receives the k-th derivative of the polynomial evaluated at x
    (res[0] is the value itself). lenres may be smaller or larger than
    lenc; extra entries beyond the polynomial's degree are set to zero.
    """
    var nmax = 0
    for i in range(lenres):
        if i < lenc:
            res[i] = c[lenc - 1]
            nmax = i
        else:
            res[i] = 0.0

    for i in range(lenc - 1):
        var k = (lenc - 1) - i
        res[0] = x * res[0] + c[k - 1]
        var lmax = nmax if nmax < k - 1 else k - 1
        for l in range(1, lmax + 1):
            res[l] = x * res[l] + res[l - 1]

    var f: Float64 = 1.0
    for i in range(2, nmax + 1):
        f *= Float64(i)
        res[i] *= f


def poly_dd_init[
    dd_origin: MutOrigin,
    mut_x: Bool,
    mut_y: Bool,
    x_origin: Origin[mut=mut_x],
    y_origin: Origin[mut=mut_y],
    //,
](
    dd: Pointer[Float64, dd_origin],
    x: Pointer[Float64, x_origin],
    y: Pointer[Float64, y_origin],
    size: Int,
):
    """Compute Newton's divided-difference coefficients for (x, y) data.

    dd must be length size. See Abramowitz & Stegun 25.2.26.
    """
    dd[0] = y[0]

    for j in range(size - 1, 0, -1):
        dd[j] = (y[j] - y[j - 1]) / (x[j] - x[j - 1])

    for i in range(2, size):
        for j in range(size - 1, i - 1, -1):
            dd[j] = (dd[j] - dd[j - 1]) / (x[j] - x[j - i])


def poly_dd_eval[
    mut_dd: Bool,
    mut_x: Bool,
    dd_origin: Origin[mut=mut_dd],
    x_origin: Origin[mut=mut_x],
    //,
](
    dd: Pointer[Float64, dd_origin],
    x: Pointer[Float64, x_origin],
    size: Int,
    xval: Float64,
) -> Float64:
    """Evaluate a divided-difference polynomial at xval."""
    var y = dd[size - 1]
    for i in range(size - 1, 0, -1):
        y = dd[i - 1] + (xval - x[i - 1]) * y
    return y


def poly_dd_taylor[
    c_origin: MutOrigin,
    mut_dd: Bool,
    mut_x: Bool,
    dd_origin: Origin[mut=mut_dd],
    x_origin: Origin[mut=mut_x],
    w_origin: MutOrigin,
    //,
](
    c: Pointer[Float64, c_origin],
    xp: Float64,
    dd: Pointer[Float64, dd_origin],
    x: Pointer[Float64, x_origin],
    size: Int,
    w: Pointer[Float64, w_origin],
):
    """Convert a divided-difference polynomial to a Taylor expansion about xp.

    c and w must both be length size.
    """
    for i in range(size):
        c[i] = 0.0
        w[i] = 0.0

    w[size - 1] = 1.0
    c[0] = dd[0]

    for i in range(size - 2, -1, -1):
        w[i] = -w[i + 1] * (x[size - 2 - i] - xp)

        for j in range(i + 1, size - 1):
            w[j] = w[j] - w[j + 1] * (x[size - 2 - i] - xp)

        for j in range(i, size):
            c[j - i] += w[j] * dd[size - i - 1]
