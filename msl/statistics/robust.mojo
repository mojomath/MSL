# SPDX-License-Identifier: GPL-3.0-or-later

# ===----------------------------------------------------------------------=== #
# MSL (Mojo Scientific Library)
#
# Derived from GNU Scientific Library (GSL)
# Original files:
#   statistics/mad_source.c
#   statistics/Sn_source.c
#   statistics/Qn_source.c
#
# Original authors:
# Copyright (C) 2018 Patrick Alken
# Copyright (C) 2005-2007 Martin Maechler, ETH Zurich
# Based on algorithms by Rousseeuw & Croux (1993)
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
Robust scale estimators (Float64): MAD, Sn, Qn.
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.math import abs
from std.memory import Pointer

# ===----------------------------------------------------------------------=== #
# MSL
# ===----------------------------------------------------------------------=== #
from msl.statistics.order import stats_median


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------


def _sort_inplace[
    o: MutOrigin,
    //,
](a: Pointer[Float64, o], n: Int):
    """Insertion sort for work arrays."""
    for i in range(1, n):
        var key = a[unsafe_offset=i]
        var j = i - 1
        while j >= 0 and a[unsafe_offset=j] > key:
            a[unsafe_offset=j + 1] = a[unsafe_offset=j]
            j -= 1
        a[unsafe_offset=j + 1] = key


# ---------------------------------------------------------------------------
# MAD
# ---------------------------------------------------------------------------


def stats_mad0[
    origin: Origin,
    work_origin: MutOrigin,
    //,
](
    data: Pointer[Float64, origin],
    stride: Int,
    n: Int,
    work: Pointer[Float64, work_origin],
) -> Float64:
    """Median absolute deviation (unscaled): median(|x_i - median(x)|).
    work must be length n."""
    for i in range(n):
        work[unsafe_offset=i] = data[unsafe_offset=i * stride]
    var median = stats_median(work, 1, n)
    for i in range(n):
        work[unsafe_offset=i] = abs(data[unsafe_offset=i * stride] - median)
    return stats_median(work, 1, n)


def stats_mad[
    origin: Origin,
    work_origin: MutOrigin,
    //,
](
    data: Pointer[Float64, origin],
    stride: Int,
    n: Int,
    work: Pointer[Float64, work_origin],
) -> Float64:
    """Median absolute deviation scaled for Gaussian consistency (MAD = 1.4826 * MAD0).
    work must be length n."""
    return 1.482602218505602 * stats_mad0(data, stride, n, work)


# ---------------------------------------------------------------------------
# Sn (Rousseeuw & Croux 1993)
# ---------------------------------------------------------------------------


def stats_Sn0_from_sorted_data[
    origin: Origin,
    work_origin: MutOrigin,
    //,
](
    sorted_data: Pointer[Float64, origin],
    stride: Int,
    n: Int,
    work: Pointer[Float64, work_origin],
) -> Float64:
    """Sn scale estimator (unscaled) from pre-sorted data. work must be length n.
    """
    var np1_2 = (n + 1) // 2

    work[unsafe_offset=0] = sorted_data[unsafe_offset=(n // 2) * stride] - sorted_data[unsafe_offset=0]

    for i in range(2, np1_2 + 1):
        var nA = i - 1
        var nB = n - i
        var diff = nB - nA
        var leftA = 1
        var leftB = 1
        var rightA = nB
        var Amin = diff // 2 + 1
        var Amax = diff // 2 + nA

        while leftA < rightA:
            var length = rightA - leftA + 1
            var even = 1 - length % 2
            var half = (length - 1) // 2
            var tryA = leftA + half
            var tryB = leftB + half
            if tryA < Amin:
                leftA = tryA + even
            elif tryA > Amax:
                rightA = tryA
                leftB = tryB + even
            else:
                var medA = (
                    sorted_data[unsafe_offset=(i - 1) * stride]
                    - sorted_data[unsafe_offset=(i - tryA + Amin - 2) * stride]
                )
                var medB = (
                    sorted_data[unsafe_offset=(tryB + i - 1) * stride]
                    - sorted_data[unsafe_offset=(i - 1) * stride]
                )
                if medA >= medB:
                    rightA = tryA
                    leftB = tryB + even
                else:
                    leftA = tryA + even

        if leftA > Amax:
            work[unsafe_offset=i - 1] = (
                sorted_data[unsafe_offset=(leftB + i - 1) * stride]
                - sorted_data[unsafe_offset=(i - 1) * stride]
            )
        else:
            var medA = (
                sorted_data[unsafe_offset=(i - 1) * stride]
                - sorted_data[unsafe_offset=(i - leftA + Amin - 2) * stride]
            )
            var medB = (
                sorted_data[unsafe_offset=(leftB + i - 1) * stride]
                - sorted_data[unsafe_offset=(i - 1) * stride]
            )
            work[unsafe_offset=i - 1] = medA if medA < medB else medB

    for i in range(np1_2 + 1, n):
        var nA = n - i
        var nB = i - 1
        var diff = nB - nA
        var leftA = 1
        var leftB = 1
        var rightA = nB
        var Amin = diff // 2 + 1
        var Amax = diff // 2 + nA

        while leftA < rightA:
            var length = rightA - leftA + 1
            var even = 1 - length % 2
            var half = (length - 1) // 2
            var tryA = leftA + half
            var tryB = leftB + half
            if tryA < Amin:
                leftA = tryA + even
            elif tryA > Amax:
                rightA = tryA
                leftB = tryB + even
            else:
                var medA = (
                    sorted_data[unsafe_offset=(i + tryA - Amin) * stride]
                    - sorted_data[unsafe_offset=(i - 1) * stride]
                )
                var medB = (
                    sorted_data[unsafe_offset=(i - 1) * stride]
                    - sorted_data[unsafe_offset=(i - tryB - 1) * stride]
                )
                if medA >= medB:
                    rightA = tryA
                    leftB = tryB + even
                else:
                    leftA = tryA + even

        if leftA > Amax:
            work[unsafe_offset=i - 1] = (
                sorted_data[unsafe_offset=(i - 1) * stride]
                - sorted_data[unsafe_offset=(i - leftB - 1) * stride]
            )
        else:
            var medA = (
                sorted_data[unsafe_offset=(i + leftA - Amin) * stride]
                - sorted_data[unsafe_offset=(i - 1) * stride]
            )
            var medB = (
                sorted_data[unsafe_offset=(i - 1) * stride]
                - sorted_data[unsafe_offset=(i - leftB - 1) * stride]
            )
            work[unsafe_offset=i - 1] = medA if medA < medB else medB

    work[unsafe_offset=n - 1] = (
        sorted_data[unsafe_offset=(n - 1) * stride] - sorted_data[unsafe_offset=(np1_2 - 1) * stride]
    )

    _sort_inplace(work, n)
    return work[unsafe_offset=np1_2 - 1]


def stats_Sn_from_sorted_data[
    origin: Origin,
    work_origin: MutOrigin,
    //,
](
    sorted_data: Pointer[Float64, origin],
    stride: Int,
    n: Int,
    work: Pointer[Float64, work_origin],
) -> Float64:
    """Sn scale estimator (sigma-consistent) from pre-sorted data. work must be length n.
    """
    var scale: Float64 = 1.1926
    var Sn0 = stats_Sn0_from_sorted_data(sorted_data, stride, n, work)
    var cn: Float64 = 1.0
    if n <= 9:
        if n == 2:
            cn = 0.743
        elif n == 3:
            cn = 1.851
        elif n == 4:
            cn = 0.954
        elif n == 5:
            cn = 1.351
        elif n == 6:
            cn = 0.993
        elif n == 7:
            cn = 1.198
        elif n == 8:
            cn = 1.005
        elif n == 9:
            cn = 1.131
    elif n % 2 == 1:
        cn = Float64(n) / (Float64(n) - 0.9)
    return scale * cn * Sn0


# ---------------------------------------------------------------------------
# Qn (Rousseeuw & Croux 1993)
# ---------------------------------------------------------------------------


def stats_Qn0_from_sorted_data[
    origin: Origin,
    work_origin: MutOrigin,
    work_int_origin: MutOrigin,
    //,
](
    sorted_data: Pointer[Float64, origin],
    stride: Int,
    n: Int,
    work: Pointer[Float64, work_origin],
    work_int: Pointer[Int, work_int_origin],
) -> Float64:
    """Qn scale estimator (unscaled) from pre-sorted data.
    work must be length 3n, work_int must be length 5n."""
    if n < 2:
        return 0.0

    var ni = n
    # var a_srt = work + n
    # var a_cand = work + 2 * n
    var a_srt = work.unsafe_offset(n)
    var a_cand = work.unsafe_offset(2 * n)
    var left = work_int
    # var right = work_int + n
    var right = work_int.unsafe_offset(n)
    # var p = work_int + 2 * n
    # var q = work_int + 3 * n
    # var weight = work_int + 4 * n
    var p = work_int.unsafe_offset(2 * n)
    var q = work_int.unsafe_offset(3 * n)
    var weight = work_int.unsafe_offset(4 * n)

    var h = n // 2 + 1
    var k = h * (h - 1) // 2

    for i in range(ni):
        left[unsafe_offset=i] = ni - i + 1
        right[unsafe_offset=i] = ni if i <= h else ni - (i - h)

    var nl = n * (n + 1) // 2
    var nr = n * n
    var knew = k + nl

    var found = False
    var trial: Float64 = 0.0

    while not found and nr - nl > ni:
        var j = 0
        for i in range(1, ni):
            if left[unsafe_offset=i] <= right[unsafe_offset=i]:
                weight[unsafe_offset=j] = right[unsafe_offset=i] - left[unsafe_offset=i] + 1
                var jh = left[unsafe_offset=i] + weight[unsafe_offset=j] // 2
                work[unsafe_offset=j] = (
                    sorted_data[unsafe_offset=i * stride] - sorted_data[unsafe_offset=(ni - jh) * stride]
                )
                j += 1

        # inline weighted high median (whimed) of work[0..j-1] with weights weight[0..j-1]
        var w_tot = 0
        for ii in range(j):
            w_tot += weight[unsafe_offset=ii]
        var wrest = 0
        var cur_j = j
        var whimed_done = False
        while not whimed_done:
            for ii in range(cur_j):
                a_srt[unsafe_offset=ii] = work[unsafe_offset=ii]
            _sort_inplace(a_srt, cur_j)
            var wh_trial = a_srt[unsafe_offset=cur_j // 2]
            var wleft = 0
            var wmid = 0
            for ii in range(cur_j):
                if work[unsafe_offset=ii] < wh_trial:
                    wleft += weight[unsafe_offset=ii]
                elif work[unsafe_offset=ii] == wh_trial:
                    wmid += weight[unsafe_offset=ii]
            var kcand = 0
            if 2 * (wrest + wleft) > w_tot:
                for ii in range(cur_j):
                    if work[unsafe_offset=ii] < wh_trial:
                        a_cand[unsafe_offset=kcand] = work[unsafe_offset=ii]
                        p[unsafe_offset=kcand] = weight[unsafe_offset=ii]
                        kcand += 1
            elif 2 * (wrest + wleft + wmid) <= w_tot:
                for ii in range(cur_j):
                    if work[unsafe_offset=ii] > wh_trial:
                        a_cand[unsafe_offset=kcand] = work[unsafe_offset=ii]
                        p[unsafe_offset=kcand] = weight[unsafe_offset=ii]
                        kcand += 1
                wrest += wleft + wmid
            else:
                trial = wh_trial
                whimed_done = True
            if not whimed_done:
                cur_j = kcand
                for ii in range(cur_j):
                    work[unsafe_offset=ii] = a_cand[unsafe_offset=ii]
                    weight[unsafe_offset=ii] = p[unsafe_offset=ii]

        j = 0
        for i in range(ni - 1, -1, -1):
            while (
                j < ni
                and sorted_data[unsafe_offset=i * stride] - sorted_data[unsafe_offset=(ni - j - 1) * stride]
                < trial
            ):
                j += 1
            p[unsafe_offset=i] = j

        j = ni + 1
        for i in range(ni):
            while (
                sorted_data[unsafe_offset=i * stride] - sorted_data[unsafe_offset=(ni - j + 1) * stride]
                > trial
            ):
                j -= 1
            q[unsafe_offset=i] = j

        var sump = 0
        var sumq = 0
        for i in range(ni):
            sump += p[unsafe_offset=i]
            sumq += q[unsafe_offset=i] - 1

        if knew <= sump:
            for i in range(ni):
                right[unsafe_offset=i] = p[unsafe_offset=i]
            nr = sump
        elif knew > sumq:
            for i in range(ni):
                left[unsafe_offset=i] = q[unsafe_offset=i]
            nl = sumq
        else:
            found = True

    if found:
        return trial

    var j = 0
    for i in range(1, ni):
        var jj = left[unsafe_offset=i]
        while jj <= right[unsafe_offset=i]:
            work[unsafe_offset=j] = sorted_data[unsafe_offset=i * stride] - sorted_data[unsafe_offset=(ni - jj) * stride]
            j += 1
            jj += 1

    knew -= nl + 1
    _sort_inplace(work, j)
    return work[unsafe_offset=knew]


def stats_Qn_from_sorted_data[
    origin: Origin,
    work_origin: MutOrigin,
    work_int_origin: MutOrigin,
    //,
](
    sorted_data: Pointer[Float64, origin],
    stride: Int,
    n: Int,
    work: Pointer[Float64, work_origin],
    work_int: Pointer[Int, work_int_origin],
) -> Float64:
    """Qn scale estimator (sigma-consistent) from pre-sorted data.
    work must be length 3n, work_int must be length 5n."""
    var scale: Float64 = 2.21914
    var Qn0 = stats_Qn0_from_sorted_data(sorted_data, stride, n, work, work_int)
    var dn: Float64 = 1.0
    if n <= 12:
        if n == 2:
            dn = 0.399356
        elif n == 3:
            dn = 0.99365
        elif n == 4:
            dn = 0.51321
        elif n == 5:
            dn = 0.84401
        elif n == 6:
            dn = 0.61220
        elif n == 7:
            dn = 0.85877
        elif n == 8:
            dn = 0.66993
        elif n == 9:
            dn = 0.87344
        elif n == 10:
            dn = 0.72014
        elif n == 11:
            dn = 0.88906
        elif n == 12:
            dn = 0.75743
    else:
        if n % 2 == 1:
            dn = 1.60188 + (-2.1284 - 5.172 / Float64(n)) / Float64(n)
        else:
            dn = 3.67561 + (
                1.9654 + (6.987 - 77.0 / Float64(n)) / Float64(n)
            ) / Float64(n)
        dn = 1.0 / (dn / Float64(n) + 1.0)
    return scale * dn * Qn0
