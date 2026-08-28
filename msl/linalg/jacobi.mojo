# SPDX-License-Identifier: GPL-3.0-or-later

# ===----------------------------------------------------------------------=== #
# MSL (Mojo Scientific Library)
#
# Derived from GNU Scientific Library (GSL)
# Original file: eigen/jacobi.c
#
# Original authors:
# Copyright (C) 2004, 2007 Brian Gough, Gerard Jungman
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
Cyclic Jacobi eigenvalue solver for real symmetric matrices (L1).

Ported from GSL's gsl_eigen_jacobi (Golub & Van Loan, Matrix
Computations, Algorithm 8.4.3). Matrices are row-major, dense, and
passed as flat Pointer buffers with an explicit leading
dimension (lda).

  A[i, j] is stored at data[i * lda + j]

`a` is destroyed during the computation. Eigenvalues are written to
`eval` (length n) and eigenvectors as columns of `evec` (n x n,
leading dimension evec_lda), which is set to the identity before
rotating.
"""

from std.math import abs, hypot

from msl.core.errno import MSL_SUCCESS, MSL_EMAXITER


def _symschur2[
    origin: MutOrigin, //,
](
    a: Pointer[Float64, origin], lda: Int, p: Int, q: Int
) -> Tuple[Float64, Float64, Float64]:
    """Return (c, s, |Apq|) for the Jacobi rotation annihilating A[p,q]."""
    var apq = a[p * lda + q]

    if apq == 0.0:
        return (1.0, 0.0, 0.0)

    var app = a[p * lda + p]
    var aqq = a[q * lda + q]
    var tau = (aqq - app) / (2.0 * apq)
    var t: Float64
    if tau >= 0.0:
        t = 1.0 / (tau + hypot(1.0, tau))
    else:
        t = -1.0 / (-tau + hypot(1.0, tau))

    var c1 = 1.0 / hypot(1.0, t)
    return (c1, t * c1, abs(apq))


def _apply_jacobi_left[
    origin: MutOrigin, //,
](
    a: Pointer[Float64, origin], lda: Int, n: Int, p: Int, q: Int, c: Float64, s: Float64
):
    for j in range(n):
        var apj = a[p * lda + j]
        var aqj = a[q * lda + j]
        a[p * lda + j] = apj * c - aqj * s
        a[q * lda + j] = apj * s + aqj * c


def _apply_jacobi_right[
    origin: MutOrigin, //,
](
    a: Pointer[Float64, origin], lda: Int, m: Int, p: Int, q: Int, c: Float64, s: Float64
):
    for i in range(m):
        var aip = a[i * lda + p]
        var aiq = a[i * lda + q]
        a[i * lda + p] = aip * c - aiq * s
        a[i * lda + q] = aip * s + aiq * c


def _offdiag_norm[
    origin: MutOrigin, //,
](a: Pointer[Float64, origin], lda: Int, n: Int) -> Float64:
    var scale: Float64 = 0.0
    var ssq: Float64 = 1.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            var aij = a[i * lda + j]
            if aij != 0.0:
                var ax = abs(aij)
                if scale < ax:
                    ssq = 1.0 + ssq * (scale / ax) * (scale / ax)
                    scale = ax
                else:
                    ssq += (ax / scale) * (ax / scale)
    return scale * (ssq**0.5)


def eigen_jacobi[
    a_origin: MutOrigin,
    eval_origin: MutOrigin,
    evec_origin: MutOrigin,
    //,
](
    a: Pointer[Float64, a_origin],
    lda: Int,
    n: Int,
    eval: Pointer[Float64, eval_origin],
    evec: Pointer[Float64, evec_origin],
    evec_lda: Int,
    max_rot: Int = 100,
) -> Int:
    """Compute eigenvalues/eigenvectors of the symmetric n x n matrix a.

    a is destroyed. eval must be length n; evec must be an n x n
    buffer (leading dimension evec_lda) and is set to the identity
    before accumulating rotations.

    Returns MSL_SUCCESS, or MSL_EMAXITER if max_rot sweeps did not
    converge (eval/evec still contain the best estimate so far).
    """
    for i in range(n):
        for j in range(n):
            evec[i * evec_lda + j] = 1.0 if i == j else 0.0

    var rot = 0
    while rot < max_rot:
        var nrm = _offdiag_norm(a, lda, n)
        if nrm == 0.0:
            break

        for p in range(n):
            for q in range(p + 1, n):
                var c: Float64
                var s: Float64
                var _red: Float64
                c, s, _red = _symschur2(a, lda, p, q)

                _apply_jacobi_left(a, lda, n, p, q, c, s)
                _apply_jacobi_right(a, lda, n, p, q, c, s)
                _apply_jacobi_right(evec, evec_lda, n, p, q, c, s)

        rot += 1

    for p in range(n):
        eval[p] = a[p * lda + p]

    if rot == max_rot:
        return MSL_EMAXITER
    return MSL_SUCCESS
