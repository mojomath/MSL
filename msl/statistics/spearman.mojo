# SPDX-License-Identifier: GPL-3.0-or-later

# ===----------------------------------------------------------------------=== #
# MSL (Mojo Scientific Library)
#
# Derived from GNU Scientific Library (GSL)
# Original files:
#   statistics/covariance_source.c
#   statistics/covariance.c
#
# Original authors:
# Copyright (C) 1996-2007 Jim Davies, Brian Gough
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
Spearman rank correlation coefficient (Float64).
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.memory import Pointer

# ===----------------------------------------------------------------------=== #
# MSL
# ===----------------------------------------------------------------------=== #
from msl.statistics.moments import stats_correlation


def _cosort[
    ox: MutOrigin,
    oy: MutOrigin,
    //,
](x: Pointer[Float64, ox], y: Pointer[Float64, oy], n: Int,):
    """Sort x ascending, dragging y along (insertion sort)."""
    for i in range(1, n):
        var kx = x[i]
        var ky = y[i]
        var j = i - 1
        while j >= 0 and x[j] > kx:
            x[j + 1] = x[j]
            y[j + 1] = y[j]
            j -= 1
        x[j + 1] = kx
        y[j + 1] = ky


def _compute_rank[
    o: MutOrigin,
    //,
](v: Pointer[Float64, o], n: Int):
    """Replace sorted v with average ranks (ties get averaged rank)."""
    var i = 0
    while i < n - 1:
        var vi = v[i]
        if vi == v[i + 1]:
            var j = i + 2
            while j < n and vi == v[j]:
                j += 1
            var rank: Float64 = 0.0
            for k in range(i, j):
                rank += Float64(k + 1)
            rank /= Float64(j - i)
            for k in range(i, j):
                v[k] = rank
            i = j
        else:
            v[i] = Float64(i + 1)
            i += 1
    if i == n - 1:
        v[n - 1] = Float64(n)


def stats_spearman[
    origin1: Origin,
    origin2: Origin,
    work_origin1: MutOrigin,
    work_origin2: MutOrigin,
    //,
](
    data1: Pointer[Float64, origin1],
    stride1: Int,
    data2: Pointer[Float64, origin2],
    stride2: Int,
    n: Int,
    ranks1: Pointer[Float64, work_origin1],
    ranks2: Pointer[Float64, work_origin2],
) -> Float64:
    """Spearman rank correlation coefficient.
    ranks1 and ranks2 are scratch buffers of length n each."""
    for i in range(n):
        ranks1[i] = data1[i * stride1]
        ranks2[i] = data2[i * stride2]

    _cosort(ranks1, ranks2, n)
    _compute_rank(ranks1, n)

    _cosort(ranks2, ranks1, n)
    _compute_rank(ranks2, n)

    return stats_correlation(ranks1, 1, ranks2, 1, n)
