# SPDX-License-Identifier: GPL-3.0-or-later

# ===----------------------------------------------------------------------=== #
# MSL (Mojo Scientific Library)
#
# Derived from GNU Scientific Library (GSL)
# Original file: linalg/lu.c
#
# Original authors:
# Copyright (C) 1996-2009 Gerard Jungman, Brian Gough
# Copyright (C) 2019, 2021 Patrick Alken
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
LU decomposition with partial pivoting for square matrices (L1).

Ported from GSL's unblocked (Level 2) LU_decomp algorithm, based on
LAPACK DGETF2. Matrices are row-major, dense, and passed as flat
Pointer buffers with an explicit leading dimension (lda).

  A[i, j] is stored at data[i * lda + j]

On output, L (unit lower triangular, diagonal not stored) occupies the
strict lower triangle and U (upper triangular) occupies the diagonal
and upper triangle of the same buffer -- i.e. decomposition is in-place.
"""

from std.math import abs, log


def lu_decomp[
    a_origin: MutOrigin,
    piv_origin: MutOrigin,
    //,
](
    a: Pointer[Float64, a_origin],
    lda: Int,
    n: Int,
    piv: Pointer[Int, piv_origin],
) -> Int:
    """Factor the n x n matrix A in-place into P A = L U.

    piv must be length n; on output piv[i] gives the row that was
    swapped with row i during elimination (raw pivot indices, not a
    full permutation array).

    Returns the sign of the permutation (+1 or -1), or 0 if the matrix
    is exactly singular.
    """
    var signum = 1
    var singular = False

    for j in range(n):
        # find pivot: largest-magnitude entry in column j, rows j..n-1
        var j_pivot = j
        var max_val = abs(a[j * lda + j])
        for i in range(j + 1, n):
            var v = abs(a[i * lda + j])
            if v > max_val:
                max_val = v
                j_pivot = i

        piv[j] = j_pivot

        var ajpj = a[j_pivot * lda + j]

        if ajpj != 0.0:
            if j_pivot != j:
                for k in range(n):
                    var tmp = a[j * lda + k]
                    a[j * lda + k] = a[j_pivot * lda + k]
                    a[j_pivot * lda + k] = tmp
                signum = -signum

            if j < n - 1:
                var ajj = a[j * lda + j]
                for i in range(j + 1, n):
                    a[i * lda + j] /= ajj
        else:
            singular = True

        if j < n - 1:
            for i in range(j + 1, n):
                var lij = a[i * lda + j]
                if lij != 0.0:
                    for k in range(j + 1, n):
                        a[i * lda + k] -= lij * a[j * lda + k]

    if singular:
        return 0
    return signum


def lu_svx[
    mut_lu: Bool,
    lu_origin: Origin[mut=mut_lu],
    mut_piv: Bool,
    piv_origin: Origin[mut=mut_piv],
    x_origin: MutOrigin,
    //,
](
    lu: Pointer[Float64, lu_origin],
    lda: Int,
    n: Int,
    piv: Pointer[Int, piv_origin],
    x: Pointer[Float64, x_origin],
):
    """Solve LU x = P b in-place, where x initially holds b.

    Applies the row swaps recorded in piv, then does forward and
    back substitution using the L and U factors packed in lu.
    """
    for j in range(n):
        var jp = piv[j]
        if jp != j:
            var tmp = x[j]
            x[j] = x[jp]
            x[jp] = tmp

    # forward substitution: L c = P b (L is unit lower triangular)
    for i in range(n):
        var sum: Float64 = 0.0
        for k in range(i):
            sum += lu[i * lda + k] * x[k]
        x[i] -= sum

    # back substitution: U x = c
    for i in range(n - 1, -1, -1):
        var sum: Float64 = 0.0
        for k in range(i + 1, n):
            sum += lu[i * lda + k] * x[k]
        x[i] = (x[i] - sum) / lu[i * lda + i]


def lu_solve[
    mut_lu: Bool,
    lu_origin: Origin[mut=mut_lu],
    mut_piv: Bool,
    piv_origin: Origin[mut=mut_piv],
    mut_b: Bool,
    b_origin: Origin[mut=mut_b],
    x_origin: MutOrigin,
    //,
](
    lu: Pointer[Float64, lu_origin],
    lda: Int,
    n: Int,
    piv: Pointer[Int, piv_origin],
    b: Pointer[Float64, b_origin],
    x: Pointer[Float64, x_origin],
):
    """Solve A x = b given the LU decomposition of A (see lu_decomp).

    x must be length n; it receives the solution. b is not modified.
    """
    for i in range(n):
        x[i] = b[i]
    lu_svx(lu, lda, n, piv, x)


def lu_det[
    mut: Bool,
    lu_origin: Origin[mut=mut],
    //,
](lu: Pointer[Float64, lu_origin], lda: Int, n: Int, signum: Int) -> Float64:
    """Compute det(A) from the LU decomposition of A."""
    var det = Float64(signum)
    for i in range(n):
        det *= lu[i * lda + i]
    return det


def lu_lndet[
    mut: Bool,
    lu_origin: Origin[mut=mut],
    //,
](lu: Pointer[Float64, lu_origin], lda: Int, n: Int) -> Float64:
    """Compute log(|det(A)|) from the LU decomposition of A."""
    var lndet: Float64 = 0.0
    for i in range(n):
        lndet += log(abs(lu[i * lda + i]))
    return lndet


def lu_invert[
    mut_lu: Bool,
    lu_origin: Origin[mut=mut_lu],
    mut_piv: Bool,
    piv_origin: Origin[mut=mut_piv],
    inv_origin: MutOrigin,
    work_origin: MutOrigin,
    //,
](
    lu: Pointer[Float64, lu_origin],
    lda: Int,
    n: Int,
    piv: Pointer[Int, piv_origin],
    inv: Pointer[Float64, inv_origin],
    inv_lda: Int,
    work: Pointer[Float64, work_origin],
):
    """Compute A^-1 from the LU decomposition of A, one column at a time.

    inv must be an n x n buffer (leading dimension inv_lda) to receive
    the inverse. work must be length n (scratch for each solve).
    """
    for col in range(n):
        for i in range(n):
            work[i] = 1.0 if i == col else 0.0
        lu_svx(lu, lda, n, piv, work)
        for i in range(n):
            inv[i * inv_lda + col] = work[i]
