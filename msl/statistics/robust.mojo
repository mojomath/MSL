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
# ===----------------------------------------------------------------------=== #
"""
Robust scale estimators (Float64): MAD, Sn, Qn.
"""

from std.math import abs
from std.memory import UnsafePointer

from .order import stats_median


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------

def _sort_inplace[o: MutOrigin, //,](a: UnsafePointer[Float64, o], n: Int):
    """Insertion sort for work arrays."""
    for i in range(1, n):
        var key = a[i]
        var j = i - 1
        while j >= 0 and a[j] > key:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = key


# ---------------------------------------------------------------------------
# MAD
# ---------------------------------------------------------------------------

def stats_mad0[
    origin: Origin,
    work_origin: MutOrigin,
    //,
](
    data: UnsafePointer[Float64, origin],
    stride: Int,
    n: Int,
    work: UnsafePointer[Float64, work_origin],
) -> Float64:
    """Median absolute deviation (unscaled): median(|x_i - median(x)|).
    work must be length n."""
    for i in range(n):
        work[i] = data[i * stride]
    var median = stats_median(work, 1, n)
    for i in range(n):
        work[i] = abs(data[i * stride] - median)
    return stats_median(work, 1, n)


def stats_mad[
    origin: Origin,
    work_origin: MutOrigin,
    //,
](
    data: UnsafePointer[Float64, origin],
    stride: Int,
    n: Int,
    work: UnsafePointer[Float64, work_origin],
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
    sorted_data: UnsafePointer[Float64, origin],
    stride: Int,
    n: Int,
    work: UnsafePointer[Float64, work_origin],
) -> Float64:
    """Sn scale estimator (unscaled) from pre-sorted data. work must be length n."""
    var np1_2 = (n + 1) // 2

    work[0] = sorted_data[(n // 2) * stride] - sorted_data[0]

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
                var medA = sorted_data[(i - 1) * stride] - sorted_data[(i - tryA + Amin - 2) * stride]
                var medB = sorted_data[(tryB + i - 1) * stride] - sorted_data[(i - 1) * stride]
                if medA >= medB:
                    rightA = tryA
                    leftB = tryB + even
                else:
                    leftA = tryA + even

        if leftA > Amax:
            work[i - 1] = sorted_data[(leftB + i - 1) * stride] - sorted_data[(i - 1) * stride]
        else:
            var medA = sorted_data[(i - 1) * stride] - sorted_data[(i - leftA + Amin - 2) * stride]
            var medB = sorted_data[(leftB + i - 1) * stride] - sorted_data[(i - 1) * stride]
            work[i - 1] = medA if medA < medB else medB

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
                var medA = sorted_data[(i + tryA - Amin) * stride] - sorted_data[(i - 1) * stride]
                var medB = sorted_data[(i - 1) * stride] - sorted_data[(i - tryB - 1) * stride]
                if medA >= medB:
                    rightA = tryA
                    leftB = tryB + even
                else:
                    leftA = tryA + even

        if leftA > Amax:
            work[i - 1] = sorted_data[(i - 1) * stride] - sorted_data[(i - leftB - 1) * stride]
        else:
            var medA = sorted_data[(i + leftA - Amin) * stride] - sorted_data[(i - 1) * stride]
            var medB = sorted_data[(i - 1) * stride] - sorted_data[(i - leftB - 1) * stride]
            work[i - 1] = medA if medA < medB else medB

    work[n - 1] = sorted_data[(n - 1) * stride] - sorted_data[(np1_2 - 1) * stride]

    _sort_inplace(work, n)
    return work[np1_2 - 1]


def stats_Sn_from_sorted_data[
    origin: Origin,
    work_origin: MutOrigin,
    //,
](
    sorted_data: UnsafePointer[Float64, origin],
    stride: Int,
    n: Int,
    work: UnsafePointer[Float64, work_origin],
) -> Float64:
    """Sn scale estimator (sigma-consistent) from pre-sorted data. work must be length n."""
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
    sorted_data: UnsafePointer[Float64, origin],
    stride: Int,
    n: Int,
    work: UnsafePointer[Float64, work_origin],
    work_int: UnsafePointer[Int, work_int_origin],
) -> Float64:
    """Qn scale estimator (unscaled) from pre-sorted data.
    work must be length 3n, work_int must be length 5n."""
    if n < 2:
        return 0.0

    var ni = n
    var a_srt = work + n
    var a_cand = work + 2 * n
    var left = work_int
    var right = work_int + n
    var p = work_int + 2 * n
    var q = work_int + 3 * n
    var weight = work_int + 4 * n

    var h = n // 2 + 1
    var k = h * (h - 1) // 2

    for i in range(ni):
        left[i] = ni - i + 1
        right[i] = ni if i <= h else ni - (i - h)

    var nl = n * (n + 1) // 2
    var nr = n * n
    var knew = k + nl

    var found = False
    var trial: Float64 = 0.0

    while not found and nr - nl > ni:
        var j = 0
        for i in range(1, ni):
            if left[i] <= right[i]:
                weight[j] = right[i] - left[i] + 1
                var jh = left[i] + weight[j] // 2
                work[j] = sorted_data[i * stride] - sorted_data[(ni - jh) * stride]
                j += 1

        # inline weighted high median (whimed) of work[0..j-1] with weights weight[0..j-1]
        var w_tot = 0
        for ii in range(j):
            w_tot += weight[ii]
        var wrest = 0
        var cur_j = j
        var whimed_done = False
        while not whimed_done:
            for ii in range(cur_j):
                a_srt[ii] = work[ii]
            _sort_inplace(a_srt, cur_j)
            var wh_trial = a_srt[cur_j // 2]
            var wleft = 0
            var wmid = 0
            for ii in range(cur_j):
                if work[ii] < wh_trial:
                    wleft += weight[ii]
                elif work[ii] == wh_trial:
                    wmid += weight[ii]
            var kcand = 0
            if 2 * (wrest + wleft) > w_tot:
                for ii in range(cur_j):
                    if work[ii] < wh_trial:
                        a_cand[kcand] = work[ii]
                        p[kcand] = weight[ii]
                        kcand += 1
            elif 2 * (wrest + wleft + wmid) <= w_tot:
                for ii in range(cur_j):
                    if work[ii] > wh_trial:
                        a_cand[kcand] = work[ii]
                        p[kcand] = weight[ii]
                        kcand += 1
                wrest += wleft + wmid
            else:
                trial = wh_trial
                whimed_done = True
            if not whimed_done:
                cur_j = kcand
                for ii in range(cur_j):
                    work[ii] = a_cand[ii]
                    weight[ii] = p[ii]

        j = 0
        for i in range(ni - 1, -1, -1):
            while j < ni and sorted_data[i * stride] - sorted_data[(ni - j - 1) * stride] < trial:
                j += 1
            p[i] = j

        j = ni + 1
        for i in range(ni):
            while sorted_data[i * stride] - sorted_data[(ni - j + 1) * stride] > trial:
                j -= 1
            q[i] = j

        var sump = 0
        var sumq = 0
        for i in range(ni):
            sump += p[i]
            sumq += q[i] - 1

        if knew <= sump:
            for i in range(ni):
                right[i] = p[i]
            nr = sump
        elif knew > sumq:
            for i in range(ni):
                left[i] = q[i]
            nl = sumq
        else:
            found = True

    if found:
        return trial

    var j = 0
    for i in range(1, ni):
        var jj = left[i]
        while jj <= right[i]:
            work[j] = sorted_data[i * stride] - sorted_data[(ni - jj) * stride]
            j += 1
            jj += 1

    knew -= nl + 1
    _sort_inplace(work, j)
    return work[knew]


def stats_Qn_from_sorted_data[
    origin: Origin,
    work_origin: MutOrigin,
    work_int_origin: MutOrigin,
    //,
](
    sorted_data: UnsafePointer[Float64, origin],
    stride: Int,
    n: Int,
    work: UnsafePointer[Float64, work_origin],
    work_int: UnsafePointer[Int, work_int_origin],
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
            dn = 3.67561 + (1.9654 + (6.987 - 77.0 / Float64(n)) / Float64(n)) / Float64(n)
        dn = 1.0 / (dn / Float64(n) + 1.0)
    return scale * dn * Qn0
