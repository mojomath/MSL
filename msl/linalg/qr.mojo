# SPDX-License-Identifier: GPL-3.0-or-later

# ===----------------------------------------------------------------------=== #
# MSL (Mojo Scientific Library)
#
# Derived from GNU Scientific Library (GSL)
# Original files:
#   linalg/qr.c
#   linalg/householder.c
#
# Original authors:
# Copyright (C) 1996-2007 Gerard Jungman, Brian Gough
# Copyright (C) 2019 Patrick Alken
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
Householder QR decomposition for square matrices (L1).

Ported from GSL's QR_decomp / householder routines. Matrices are
row-major, dense, and passed as flat Pointer buffers with an
explicit leading dimension (lda).

  A[i, j] is stored at data[i * lda + j]

On output, R (upper triangular) occupies the diagonal and upper
triangle of the buffer. The Householder vectors that implicitly
represent Q are packed into the strict lower triangle (element [i,i]
of each Householder vector is implicitly 1 and not stored); tau[i]
holds the corresponding Householder coefficients.
"""

from std.math import hypot, sqrt


def _sign(x: Float64) -> Float64:
    return 1.0 if x >= 0.0 else -1.0


def qr_decomp[
    a_origin: MutOrigin,
    tau_origin: MutOrigin,
    //,
](a: Pointer[Float64, a_origin], lda: Int, n: Int, tau: Pointer[Float64, tau_origin]):
    """Factor the n x n matrix A in-place into A = Q R.

    tau must be length n.
    """
    for i in range(n):
        # compute Householder transform of column i, rows i..n-1
        var alpha = a[i * lda + i]
        var xnorm_sq: Float64 = 0.0
        for k in range(i + 1, n):
            var v = a[k * lda + i]
            xnorm_sq += v * v
        var xnorm = sqrt(xnorm_sq)

        if xnorm == 0.0:
            tau[i] = 0.0
        else:
            var beta = -_sign(alpha) * hypot(alpha, xnorm)
            var tau_i = (beta - alpha) / beta
            tau[i] = tau_i

            var s = alpha - beta
            for k in range(i + 1, n):
                a[k * lda + i] /= s
            a[i * lda + i] = beta

            # apply the reflector to the remaining columns i+1..n-1
            if i + 1 < n:
                for j in range(i + 1, n):
                    # d = v' * A[:,j] over rows i..n-1, with v[i] implicitly 1
                    var d = a[i * lda + j]
                    for k in range(i + 1, n):
                        d += a[k * lda + i] * a[k * lda + j]
                    a[i * lda + j] -= tau_i * d
                    for k in range(i + 1, n):
                        a[k * lda + j] -= tau_i * d * a[k * lda + i]


def _qr_qtvec[
    mut_qr: Bool,
    qr_origin: Origin[mut=mut_qr],
    mut_tau: Bool,
    tau_origin: Origin[mut=mut_tau],
    v_origin: MutOrigin,
    //,
](
    qr: Pointer[Float64, qr_origin],
    lda: Int,
    n: Int,
    tau: Pointer[Float64, tau_origin],
    v: Pointer[Float64, v_origin],
):
    """Compute v <- Q^T v in-place, given the packed QR factorization."""
    for i in range(n):
        var tau_i = tau[i]
        if tau_i == 0.0:
            continue

        var w0 = v[i]
        var d1: Float64 = 0.0
        for k in range(i + 1, n):
            d1 += qr[k * lda + i] * v[k]
        var d = w0 + d1

        v[i] = w0 - tau_i * d
        for k in range(i + 1, n):
            v[k] -= tau_i * d * qr[k * lda + i]


def qr_svx[
    mut_qr: Bool,
    qr_origin: Origin[mut=mut_qr],
    mut_tau: Bool,
    tau_origin: Origin[mut=mut_tau],
    x_origin: MutOrigin,
    //,
](
    qr: Pointer[Float64, qr_origin],
    lda: Int,
    n: Int,
    tau: Pointer[Float64, tau_origin],
    x: Pointer[Float64, x_origin],
):
    """Solve R x = Q^T b in-place, where x initially holds b."""
    _qr_qtvec(qr, lda, n, tau, x)

    for i in range(n - 1, -1, -1):
        var sum = x[i]
        for k in range(i + 1, n):
            sum -= qr[i * lda + k] * x[k]
        x[i] = sum / qr[i * lda + i]


def qr_solve[
    mut_qr: Bool,
    qr_origin: Origin[mut=mut_qr],
    mut_tau: Bool,
    tau_origin: Origin[mut=mut_tau],
    mut_b: Bool,
    b_origin: Origin[mut=mut_b],
    x_origin: MutOrigin,
    //,
](
    qr: Pointer[Float64, qr_origin],
    lda: Int,
    n: Int,
    tau: Pointer[Float64, tau_origin],
    b: Pointer[Float64, b_origin],
    x: Pointer[Float64, x_origin],
):
    """Solve A x = b given the QR decomposition of A (see qr_decomp).

    x must be length n; b is not modified.
    """
    for i in range(n):
        x[i] = b[i]
    qr_svx(qr, lda, n, tau, x)
