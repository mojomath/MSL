# SPDX-License-Identifier: GPL-3.0-or-later

# ===----------------------------------------------------------------------=== #
# MSL (Mojo Scientific Library)
#
# Derived from GNU Scientific Library (GSL)
# Original files:
#   statistics/minmax_source.c
#   statistics/select_source.c
#   statistics/median_source.c
#   statistics/quantiles_source.c
#   statistics/trmean_source.c
#   statistics/gastwirth_source.c
#
# Original authors:
# Copyright (C) 1996-2007 Jim Davies, Brian Gough
# Copyright (C) 2018 Patrick Alken
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
Order statistics, quantiles, and robust location estimators (Float64),
ported from GSL statistics module.
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.math import floor
from std.memory import Pointer


# ---------------------------------------------------------------------------
# min / max
# ---------------------------------------------------------------------------


def stats_max[
    origin: Origin
](data: Pointer[Float64, origin], stride: Int, n: Int) -> Float64:
    """Largest element of the dataset."""
    var result = data[unsafe_offset=0]
    for i in range(n):
        var xi = data[unsafe_offset=i * stride]
        if xi > result:
            result = xi
    return result


def stats_min[
    origin: Origin
](data: Pointer[Float64, origin], stride: Int, n: Int) -> Float64:
    """Smallest element of the dataset."""
    var result = data[unsafe_offset=0]
    for i in range(n):
        var xi = data[unsafe_offset=i * stride]
        if xi < result:
            result = xi
    return result


def stats_minmax[
    origin: Origin
](data: Pointer[Float64, origin], stride: Int, n: Int) -> Tuple[
    Float64, Float64
]:
    """Returns (min, max) of the dataset."""
    var mn = data[unsafe_offset=0]
    var mx = data[unsafe_offset=0]
    for i in range(n):
        var xi = data[unsafe_offset=i * stride]
        if xi < mn:
            mn = xi
        if xi > mx:
            mx = xi
    return (mn, mx)


# ---------------------------------------------------------------------------
# min / max index
# ---------------------------------------------------------------------------


def stats_max_index[
    origin: Origin
](data: Pointer[Float64, origin], stride: Int, n: Int) -> Int:
    """Index of the largest element (first occurrence)."""
    var mx = data[unsafe_offset=0]
    var idx = 0
    for i in range(n):
        var xi = data[unsafe_offset=i * stride]
        if xi > mx:
            mx = xi
            idx = i
    return idx


def stats_min_index[
    origin: Origin
](data: Pointer[Float64, origin], stride: Int, n: Int) -> Int:
    """Index of the smallest element (first occurrence)."""
    var mn = data[unsafe_offset=0]
    var idx = 0
    for i in range(n):
        var xi = data[unsafe_offset=i * stride]
        if xi < mn:
            mn = xi
            idx = i
    return idx


def stats_minmax_index[
    origin: Origin
](data: Pointer[Float64, origin], stride: Int, n: Int) -> Tuple[Int, Int]:
    """Returns (min_index, max_index) of the dataset."""
    var mn = data[unsafe_offset=0]
    var mx = data[unsafe_offset=0]
    var min_idx = 0
    var max_idx = 0
    for i in range(n):
        var xi = data[unsafe_offset=i * stride]
        if xi < mn:
            mn = xi
            min_idx = i
        if xi > mx:
            mx = xi
            max_idx = i
    return (min_idx, max_idx)


# ---------------------------------------------------------------------------
# quickselect (mutates data)
# ---------------------------------------------------------------------------


def stats_select[
    origin: MutOrigin
](data: Pointer[Float64, origin], stride: Int, n: Int, k: Int) -> Float64:
    """K-th smallest element via quickselect (partially sorts data in-place)."""
    var left = 0
    var right = n - 1

    while True:
        if right <= left + 1:
            if right == left + 1 and data[unsafe_offset=right * stride] < data[unsafe_offset=left * stride]:
                var tmp = data[unsafe_offset=left * stride]
                data[unsafe_offset=left * stride] = data[unsafe_offset=right * stride]
                data[unsafe_offset=right * stride] = tmp
            return data[unsafe_offset=k * stride]

        var mid = (left + right) >> 1

        var tmp = data[unsafe_offset=mid * stride]
        data[unsafe_offset=mid * stride] = data[unsafe_offset=(left + 1) * stride]
        data[unsafe_offset=(left + 1) * stride] = tmp

        if data[unsafe_offset=left * stride] > data[unsafe_offset=right * stride]:
            tmp = data[unsafe_offset=left * stride]
            data[unsafe_offset=left * stride] = data[unsafe_offset=right * stride]
            data[unsafe_offset=right * stride] = tmp

        if data[unsafe_offset=(left + 1) * stride] > data[unsafe_offset=right * stride]:
            tmp = data[unsafe_offset=(left + 1) * stride]
            data[unsafe_offset=(left + 1) * stride] = data[unsafe_offset=right * stride]
            data[unsafe_offset=right * stride] = tmp

        if data[unsafe_offset=left * stride] > data[unsafe_offset=(left + 1) * stride]:
            tmp = data[unsafe_offset=left * stride]
            data[unsafe_offset=left * stride] = data[unsafe_offset=(left + 1) * stride]
            data[unsafe_offset=(left + 1) * stride] = tmp

        var i = left + 1
        var j = right
        var pivot = data[unsafe_offset=(left + 1) * stride]

        while True:
            i += 1
            while data[unsafe_offset=i * stride] < pivot:
                i += 1
            j -= 1
            while data[unsafe_offset=j * stride] > pivot:
                j -= 1
            if j < i:
                break
            tmp = data[unsafe_offset=i * stride]
            data[unsafe_offset=i * stride] = data[unsafe_offset=j * stride]
            data[unsafe_offset=j * stride] = tmp

        data[unsafe_offset=(left + 1) * stride] = data[unsafe_offset=j * stride]
        data[unsafe_offset=j * stride] = pivot

        if j >= k:
            right = j - 1
        if j <= k:
            left = i
        if j == k:
            return data[unsafe_offset=k * stride]


# ---------------------------------------------------------------------------
# median
# ---------------------------------------------------------------------------


def stats_median_from_sorted_data[
    origin: Origin
](sorted_data: Pointer[Float64, origin], stride: Int, n: Int) -> Float64:
    """Median of pre-sorted data."""
    if n == 0:
        return 0.0
    var lhs = (n - 1) // 2
    var rhs = n // 2
    if lhs == rhs:
        return sorted_data[unsafe_offset=lhs * stride]
    return (sorted_data[unsafe_offset=lhs * stride] + sorted_data[unsafe_offset=rhs * stride]) * 0.5


def stats_median[
    origin: MutOrigin
](data: Pointer[Float64, origin], stride: Int, n: Int) -> Float64:
    """Median of unsorted data (partially sorts data in-place via quickselect).
    """
    if n == 0:
        return 0.0
    var lhs = (n - 1) // 2
    var rhs = n // 2
    if lhs == rhs:
        return stats_select(data, stride, n, lhs)
    var a = stats_select(data, stride, n, lhs)
    var b = stats_select(data, stride, n, rhs)
    return 0.5 * (a + b)


# ---------------------------------------------------------------------------
# quantile
# ---------------------------------------------------------------------------


def stats_quantile_from_sorted_data[
    origin: Origin
](
    sorted_data: Pointer[Float64, origin], stride: Int, n: Int, f: Float64
) -> Float64:
    """Quantile at fraction f in [0,1] from pre-sorted data (linear interpolation).
    """
    if n == 0:
        return 0.0
    var index = f * Float64(n - 1)
    var lhs = Int(index)
    var delta = index - Float64(lhs)
    if lhs == n - 1:
        return sorted_data[unsafe_offset=lhs * stride]
    return (1.0 - delta) * sorted_data[unsafe_offset=lhs * stride] + delta * sorted_data[unsafe_offset=
        (lhs + 1) * stride
    ]


# ---------------------------------------------------------------------------
# trimmed mean
# ---------------------------------------------------------------------------


def stats_trmean_from_sorted_data[
    origin: Origin
](
    trim: Float64,
    sorted_data: Pointer[Float64, origin],
    stride: Int,
    n: Int,
) -> Float64:
    """Trimmed mean: drop `trim` fraction from each tail of sorted data."""
    if trim >= 0.5:
        return stats_median_from_sorted_data(sorted_data, stride, n)
    var ilow = Int(floor(trim * Float64(n)))
    var ihigh = n - ilow - 1
    var mean: Float64 = 0.0
    var k: Float64 = 0.0
    for i in range(ilow, ihigh + 1):
        var delta = sorted_data[unsafe_offset=i * stride] - mean
        k += 1.0
        mean += delta / k
    return mean


# ---------------------------------------------------------------------------
# Gastwirth location estimator
# ---------------------------------------------------------------------------


def stats_gastwirth_from_sorted_data[
    origin: Origin
](sorted_data: Pointer[Float64, origin], stride: Int, n: Int) -> Float64:
    """Gastwirth's location estimator: 0.3*Q(1/3) + 0.4*median + 0.3*Q(2/3)."""
    if n == 0:
        return 0.0
    var a = stats_quantile_from_sorted_data(sorted_data, stride, n, 1.0 / 3.0)
    var b = stats_median_from_sorted_data(sorted_data, stride, n)
    var c = stats_quantile_from_sorted_data(sorted_data, stride, n, 2.0 / 3.0)
    return 0.3 * a + 0.4 * b + 0.3 * c
