# SPDX-License-Identifier: GPL-3.0-or-later

# ===----------------------------------------------------------------------=== #
# MSL (Mojo Scientific Library)
#
# Derived from GNU Scientific Library (GSL)
# Original files:
#   statistics/mean_source.c
#   statistics/variance_source.c
#   statistics/absdev_source.c
#   statistics/skew_source.c
#   statistics/kurtosis_source.c
#   statistics/lag1_source.c
#   statistics/covariance_source.c
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
Core descriptive statistics (Float64), ported from GSL statistics module.
"""

from std.math import abs, sqrt
from std.memory import Pointer


def _compute_variance[
    origin: Origin
](
    data: Pointer[Float64, origin], stride: Int, n: Int, mean: Float64
) -> Float64:
    var variance: Float64 = 0.0
    for i in range(n):
        var delta = data[i * stride] - mean
        variance += (delta * delta - variance) / Float64(i + 1)
    return variance


def _compute_tss[
    origin: Origin
](
    data: Pointer[Float64, origin], stride: Int, n: Int, mean: Float64
) -> Float64:
    var tss: Float64 = 0.0
    for i in range(n):
        var delta = data[i * stride] - mean
        tss += delta * delta
    return tss


def _compute_covariance[
    origin1: Origin, origin2: Origin
](
    data1: Pointer[Float64, origin1],
    stride1: Int,
    data2: Pointer[Float64, origin2],
    stride2: Int,
    n: Int,
    mean1: Float64,
    mean2: Float64,
) -> Float64:
    var covariance: Float64 = 0.0
    for i in range(n):
        var delta1 = data1[i * stride1] - mean1
        var delta2 = data2[i * stride2] - mean2
        covariance += (delta1 * delta2 - covariance) / Float64(i + 1)
    return covariance


def stats_mean[
    origin: Origin
](data: Pointer[Float64, origin], stride: Int, n: Int) -> Float64:
    """Arithmetic mean."""
    if n <= 0:
        return 0.0

    var mean: Float64 = 0.0
    for i in range(n):
        mean += (data[i * stride] - mean) / Float64(i + 1)
    return mean


def stats_variance_with_fixed_mean[
    origin: Origin
](
    data: Pointer[Float64, origin], stride: Int, n: Int, mean: Float64
) -> Float64:
    """Variance about a fixed mean (divides by n)."""
    if n <= 0:
        return 0.0
    return _compute_variance(data, stride, n, mean)


def stats_sd_with_fixed_mean[
    origin: Origin
](
    data: Pointer[Float64, origin], stride: Int, n: Int, mean: Float64
) -> Float64:
    """Standard deviation about a fixed mean (divides by n)."""
    return sqrt(stats_variance_with_fixed_mean(data, stride, n, mean))


def stats_variance_m[
    origin: Origin
](
    data: Pointer[Float64, origin], stride: Int, n: Int, mean: Float64
) -> Float64:
    """Sample variance about provided mean (divides by n-1)."""
    if n <= 1:
        return 0.0
    return _compute_variance(data, stride, n, mean) * (
        Float64(n) / Float64(n - 1)
    )


def stats_sd_m[
    origin: Origin
](
    data: Pointer[Float64, origin], stride: Int, n: Int, mean: Float64
) -> Float64:
    """Sample standard deviation about provided mean."""
    return sqrt(stats_variance_m(data, stride, n, mean))


def stats_variance[
    origin: Origin
](data: Pointer[Float64, origin], stride: Int, n: Int) -> Float64:
    """Sample variance."""
    var mean = stats_mean(data, stride, n)
    return stats_variance_m(data, stride, n, mean)


def stats_sd[
    origin: Origin
](data: Pointer[Float64, origin], stride: Int, n: Int) -> Float64:
    """Sample standard deviation."""
    var mean = stats_mean(data, stride, n)
    return stats_sd_m(data, stride, n, mean)


def stats_tss_m[
    origin: Origin
](
    data: Pointer[Float64, origin], stride: Int, n: Int, mean: Float64
) -> Float64:
    """Total sum of squares around provided mean."""
    if n <= 0:
        return 0.0
    return _compute_tss(data, stride, n, mean)


def stats_tss[
    origin: Origin
](data: Pointer[Float64, origin], stride: Int, n: Int) -> Float64:
    """Total sum of squares around sample mean."""
    var mean = stats_mean(data, stride, n)
    return stats_tss_m(data, stride, n, mean)


def stats_absdev[
    origin: Origin
](data: Pointer[Float64, origin], stride: Int, n: Int) -> Float64:
    """Mean absolute deviation."""
    var mean = stats_mean(data, stride, n)
    return stats_absdev_m(data, stride, n, mean)


def stats_absdev_m[
    origin: Origin
](
    data: Pointer[Float64, origin], stride: Int, n: Int, mean: Float64
) -> Float64:
    """Mean absolute deviation about provided mean."""
    if n <= 0:
        return 0.0

    var sum: Float64 = 0.0
    for i in range(n):
        sum += abs(data[i * stride] - mean)
    return sum / Float64(n)


def stats_skew[
    origin: Origin
](data: Pointer[Float64, origin], stride: Int, n: Int) -> Float64:
    """Skewness."""
    var mean = stats_mean(data, stride, n)
    var sd = stats_sd_m(data, stride, n, mean)
    return stats_skew_m_sd(data, stride, n, mean, sd)


def stats_skew_m_sd[
    origin: Origin
](
    data: Pointer[Float64, origin],
    stride: Int,
    n: Int,
    mean: Float64,
    sd: Float64,
) -> Float64:
    """Skewness about provided mean and sd."""
    if n <= 0 or sd == 0.0:
        return 0.0

    var skew: Float64 = 0.0
    for i in range(n):
        var x = (data[i * stride] - mean) / sd
        skew += (x * x * x - skew) / Float64(i + 1)
    return skew


def stats_kurtosis[
    origin: Origin
](data: Pointer[Float64, origin], stride: Int, n: Int) -> Float64:
    """Excess kurtosis (0 for Gaussian)."""
    var mean = stats_mean(data, stride, n)
    var sd = stats_sd_m(data, stride, n, mean)
    return stats_kurtosis_m_sd(data, stride, n, mean, sd)


def stats_kurtosis_m_sd[
    origin: Origin
](
    data: Pointer[Float64, origin],
    stride: Int,
    n: Int,
    mean: Float64,
    sd: Float64,
) -> Float64:
    """Excess kurtosis about provided mean and sd."""
    if n <= 0 or sd == 0.0:
        return 0.0

    var avg: Float64 = 0.0
    for i in range(n):
        var x = (data[i * stride] - mean) / sd
        avg += (x * x * x * x - avg) / Float64(i + 1)
    return avg - 3.0


def stats_lag1_autocorrelation[
    origin: Origin
](data: Pointer[Float64, origin], stride: Int, n: Int) -> Float64:
    """Lag-1 autocorrelation."""
    var mean = stats_mean(data, stride, n)
    return stats_lag1_autocorrelation_m(data, stride, n, mean)


def stats_lag1_autocorrelation_m[
    origin: Origin
](
    data: Pointer[Float64, origin], stride: Int, n: Int, mean: Float64
) -> Float64:
    """Lag-1 autocorrelation about provided mean."""
    if n <= 1:
        return 0.0

    var q: Float64 = 0.0
    var d0 = data[0] - mean
    var v: Float64 = d0 * d0

    for i in range(1, n):
        var delta0 = data[(i - 1) * stride] - mean
        var delta1 = data[i * stride] - mean
        q += (delta0 * delta1 - q) / Float64(i + 1)
        v += (delta1 * delta1 - v) / Float64(i + 1)

    if v == 0.0:
        return 0.0
    return q / v


def stats_covariance_m[
    origin1: Origin, origin2: Origin
](
    data1: Pointer[Float64, origin1],
    stride1: Int,
    data2: Pointer[Float64, origin2],
    stride2: Int,
    n: Int,
    mean1: Float64,
    mean2: Float64,
) -> Float64:
    """Sample covariance using provided means."""
    if n <= 1:
        return 0.0
    var c = _compute_covariance(data1, stride1, data2, stride2, n, mean1, mean2)
    return c * (Float64(n) / Float64(n - 1))


def stats_covariance[
    origin1: Origin, origin2: Origin
](
    data1: Pointer[Float64, origin1],
    stride1: Int,
    data2: Pointer[Float64, origin2],
    stride2: Int,
    n: Int,
) -> Float64:
    """Sample covariance."""
    var mean1 = stats_mean(data1, stride1, n)
    var mean2 = stats_mean(data2, stride2, n)
    return stats_covariance_m(data1, stride1, data2, stride2, n, mean1, mean2)


def stats_correlation[
    origin1: Origin, origin2: Origin
](
    data1: Pointer[Float64, origin1],
    stride1: Int,
    data2: Pointer[Float64, origin2],
    stride2: Int,
    n: Int,
) -> Float64:
    """Pearson correlation coefficient using Welford recurrence."""
    if n <= 1:
        return 0.0

    var sum_xsq: Float64 = 0.0
    var sum_ysq: Float64 = 0.0
    var sum_cross: Float64 = 0.0

    var mean_x = data1[0]
    var mean_y = data2[0]

    for i in range(1, n):
        var ratio = Float64(i) / Float64(i + 1)
        var delta_x = data1[i * stride1] - mean_x
        var delta_y = data2[i * stride2] - mean_y
        sum_xsq += delta_x * delta_x * ratio
        sum_ysq += delta_y * delta_y * ratio
        sum_cross += delta_x * delta_y * ratio
        mean_x += delta_x / Float64(i + 1)
        mean_y += delta_y / Float64(i + 1)

    var denom = sqrt(sum_xsq) * sqrt(sum_ysq)
    if denom == 0.0:
        return 0.0
    return sum_cross / denom
