# SPDX-License-Identifier: GPL-3.0-or-later

# ===----------------------------------------------------------------------=== #
# MSL (Mojo Scientific Library)
#
# Derived from GNU Scientific Library (GSL)
# Original files:
#   statistics/p_variance_source.c
#   statistics/ttest_source.c
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
# ===----------------------------------------------------------------------=== #
"""
Two-sample inferential statistics (Float64), ported from GSL statistics module.
"""

from std.math import sqrt
from std.memory import UnsafePointer

from .moments import stats_mean, stats_variance


def stats_pvariance[
    origin1: Origin, origin2: Origin
](
    data1: UnsafePointer[Float64, origin1],
    stride1: Int,
    n1: Int,
    data2: UnsafePointer[Float64, origin2],
    stride2: Int,
    n2: Int,
) -> Float64:
    """Pooled variance of two independent samples."""
    var var1 = stats_variance(data1, stride1, n1)
    var var2 = stats_variance(data2, stride2, n2)
    return (Float64(n1 - 1) * var1 + Float64(n2 - 1) * var2) / Float64(
        n1 + n2 - 2
    )


def stats_ttest[
    origin1: Origin, origin2: Origin
](
    data1: UnsafePointer[Float64, origin1],
    stride1: Int,
    n1: Int,
    data2: UnsafePointer[Float64, origin2],
    stride2: Int,
    n2: Int,
) -> Float64:
    """Two-sample independent t-statistic (equal-variance / pooled)."""
    var mean1 = stats_mean(data1, stride1, n1)
    var mean2 = stats_mean(data2, stride2, n2)
    var pv = stats_pvariance(data1, stride1, n1, data2, stride2, n2)
    return (mean1 - mean2) / sqrt(pv * (1.0 / Float64(n1) + 1.0 / Float64(n2)))
