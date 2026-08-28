# SPDX-License-Identifier: GPL-3.0-or-later

# ===----------------------------------------------------------------------=== #
# MSL (Mojo Scientific Library)
#
# Derived from GNU Scientific Library (GSL)
# Original file: linalg/cholesky.c
#
# Original authors:
# Copyright (C) 1996-2007 Gerard Jungman, Brian Gough
# Copyright (C) 2016, 2019 Patrick Alken
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
Cholesky decomposition for symmetric positive-definite matrices (L1).

Ported from GSL's unblocked (Level 2) Cholesky algorithm. Matrices are
row-major, dense, and passed as flat Pointer buffers with an
explicit leading dimension (lda).

  A[i, j] is stored at data[i * lda + j]

On output, the lower triangle (including diagonal) of the buffer holds
L such that A = L L^T. The strict upper triangle is zeroed.
"""

from std.math import sqrt

from msl.core.errno import MSL_SUCCESS, MSL_EDOM


def cholesky_decomp[
    a_origin: MutOrigin,
    //,
](a: Pointer[Float64, a_origin], lda: Int, n: Int) -> Int:
    """Factor the symmetric positive-definite n x n matrix A in-place
    into A = L L^T.

    Returns MSL_SUCCESS, or MSL_EDOM if A is not positive-definite.
    """
    for j in range(n):
        var sum = a[j * lda + j]
        for k in range(j):
            sum -= a[j * lda + k] * a[j * lda + k]

        if sum <= 0.0:
            return MSL_EDOM

        var ljj = sqrt(sum)
        a[j * lda + j] = ljj

        for i in range(j + 1, n):
            var s = a[i * lda + j]
            for k in range(j):
                s -= a[i * lda + k] * a[j * lda + k]
            a[i * lda + j] = s / ljj

    for i in range(n):
        for j in range(i + 1, n):
            a[i * lda + j] = 0.0

    return MSL_SUCCESS


def cholesky_svx[
    mut_l: Bool,
    l_origin: Origin[mut=mut_l],
    x_origin: MutOrigin,
    //,
](l: Pointer[Float64, l_origin], lda: Int, n: Int, x: Pointer[Float64, x_origin]):
    """Solve L L^T x = b in-place, where x initially holds b."""
    # forward substitution: L c = b
    for i in range(n):
        var sum = x[i]
        for k in range(i):
            sum -= l[i * lda + k] * x[k]
        x[i] = sum / l[i * lda + i]

    # back substitution: L^T x = c
    for i in range(n - 1, -1, -1):
        var sum = x[i]
        for k in range(i + 1, n):
            sum -= l[k * lda + i] * x[k]
        x[i] = sum / l[i * lda + i]


def cholesky_solve[
    mut_l: Bool,
    l_origin: Origin[mut=mut_l],
    mut_b: Bool,
    b_origin: Origin[mut=mut_b],
    x_origin: MutOrigin,
    //,
](
    l: Pointer[Float64, l_origin],
    lda: Int,
    n: Int,
    b: Pointer[Float64, b_origin],
    x: Pointer[Float64, x_origin],
):
    """Solve A x = b given the Cholesky decomposition of A (see
    cholesky_decomp). x must be length n; b is not modified."""
    for i in range(n):
        x[i] = b[i]
    cholesky_svx(l, lda, n, x)
