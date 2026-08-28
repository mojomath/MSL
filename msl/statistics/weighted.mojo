# SPDX-License-Identifier: GPL-3.0-or-later

# ===----------------------------------------------------------------------=== #
# MSL (Mojo Scientific Library)
#
# Derived from GNU Scientific Library (GSL)
# Original files:
#   statistics/wmean_source.c
#   statistics/wvariance_source.c
#   statistics/wabsdev_source.c
#   statistics/wskew_source.c
#   statistics/wkurtosis_source.c
#
# Original authors:
# Copyright (C) 1996-2010 Jim Davies, Brian Gough
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
Weighted descriptive statistics (Float64), ported from GSL statistics.
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.math import (
    abs,
    sqrt,
)
from std.memory import Pointer


def stats_wmean[
    w_origin: Origin, x_origin: Origin
](
    w: Pointer[Float64, w_origin],
    wstride: Int,
    data: Pointer[Float64, x_origin],
    stride: Int,
    n: Int,
) -> Float64:
    """Weighted arithmetic mean (positive weights only)."""
    var wmean: Float64 = 0.0
    var W: Float64 = 0.0
    for i in range(n):
        var wi = w[unsafe_offset=i * wstride]
        if wi > 0.0:
            W += wi
            wmean += (data[unsafe_offset=i * stride] - wmean) * (wi / W)
    return wmean


def _compute_wvariance[
    w_origin: Origin, x_origin: Origin
](
    w: Pointer[Float64, w_origin],
    wstride: Int,
    data: Pointer[Float64, x_origin],
    stride: Int,
    n: Int,
    wmean: Float64,
) -> Float64:
    var wvariance: Float64 = 0.0
    var W: Float64 = 0.0
    for i in range(n):
        var wi = w[unsafe_offset=i * wstride]
        if wi > 0.0:
            var delta = data[unsafe_offset=i * stride] - wmean
            W += wi
            wvariance += (delta * delta - wvariance) * (wi / W)
    return wvariance


def _compute_wtss[
    w_origin: Origin, x_origin: Origin
](
    w: Pointer[Float64, w_origin],
    wstride: Int,
    data: Pointer[Float64, x_origin],
    stride: Int,
    n: Int,
    wmean: Float64,
) -> Float64:
    var wtss: Float64 = 0.0
    for i in range(n):
        var wi = w[unsafe_offset=i * wstride]
        if wi > 0.0:
            var delta = data[unsafe_offset=i * stride] - wmean
            wtss += wi * delta * delta
    return wtss


def _compute_weight_scale[
    w_origin: Origin
](w: Pointer[Float64, w_origin], wstride: Int, n: Int) -> Float64:
    var a: Float64 = 0.0
    var b: Float64 = 0.0
    for i in range(n):
        var wi = w[unsafe_offset=i * wstride]
        if wi > 0.0:
            a += wi
            b += wi * wi
    var denom = (a * a) - b
    if denom == 0.0:
        return 0.0
    return (a * a) / denom


def stats_wvariance_with_fixed_mean[
    w_origin: Origin, x_origin: Origin
](
    w: Pointer[Float64, w_origin],
    wstride: Int,
    data: Pointer[Float64, x_origin],
    stride: Int,
    n: Int,
    wmean: Float64,
) -> Float64:
    """Weighted variance around provided mean."""
    return _compute_wvariance(w, wstride, data, stride, n, wmean)


def stats_wsd_with_fixed_mean[
    w_origin: Origin, x_origin: Origin
](
    w: Pointer[Float64, w_origin],
    wstride: Int,
    data: Pointer[Float64, x_origin],
    stride: Int,
    n: Int,
    wmean: Float64,
) -> Float64:
    """Weighted sd around provided mean."""
    return sqrt(
        stats_wvariance_with_fixed_mean(w, wstride, data, stride, n, wmean)
    )


def stats_wvariance_m[
    w_origin: Origin, x_origin: Origin
](
    w: Pointer[Float64, w_origin],
    wstride: Int,
    data: Pointer[Float64, x_origin],
    stride: Int,
    n: Int,
    wmean: Float64,
) -> Float64:
    """Weighted sample variance (unbiased scaling)."""
    var v = _compute_wvariance(w, wstride, data, stride, n, wmean)
    var scale = _compute_weight_scale(w, wstride, n)
    return scale * v


def stats_wsd_m[
    w_origin: Origin, x_origin: Origin
](
    w: Pointer[Float64, w_origin],
    wstride: Int,
    data: Pointer[Float64, x_origin],
    stride: Int,
    n: Int,
    wmean: Float64,
) -> Float64:
    """Weighted sample standard deviation."""
    return sqrt(stats_wvariance_m(w, wstride, data, stride, n, wmean))


def stats_wvariance[
    w_origin: Origin, x_origin: Origin
](
    w: Pointer[Float64, w_origin],
    wstride: Int,
    data: Pointer[Float64, x_origin],
    stride: Int,
    n: Int,
) -> Float64:
    var wmean = stats_wmean(w, wstride, data, stride, n)
    return stats_wvariance_m(w, wstride, data, stride, n, wmean)


def stats_wsd[
    w_origin: Origin, x_origin: Origin
](
    w: Pointer[Float64, w_origin],
    wstride: Int,
    data: Pointer[Float64, x_origin],
    stride: Int,
    n: Int,
) -> Float64:
    var wmean = stats_wmean(w, wstride, data, stride, n)
    return stats_wsd_m(w, wstride, data, stride, n, wmean)


def stats_wtss_m[
    w_origin: Origin, x_origin: Origin
](
    w: Pointer[Float64, w_origin],
    wstride: Int,
    data: Pointer[Float64, x_origin],
    stride: Int,
    n: Int,
    wmean: Float64,
) -> Float64:
    return _compute_wtss(w, wstride, data, stride, n, wmean)


def stats_wtss[
    w_origin: Origin, x_origin: Origin
](
    w: Pointer[Float64, w_origin],
    wstride: Int,
    data: Pointer[Float64, x_origin],
    stride: Int,
    n: Int,
) -> Float64:
    var wmean = stats_wmean(w, wstride, data, stride, n)
    return stats_wtss_m(w, wstride, data, stride, n, wmean)


def stats_wabsdev_m[
    w_origin: Origin, x_origin: Origin
](
    w: Pointer[Float64, w_origin],
    wstride: Int,
    data: Pointer[Float64, x_origin],
    stride: Int,
    n: Int,
    wmean: Float64,
) -> Float64:
    """Weighted mean absolute deviation."""
    var wabsdev: Float64 = 0.0
    var W: Float64 = 0.0
    for i in range(n):
        var wi = w[unsafe_offset=i * wstride]
        if wi > 0.0:
            var delta = abs(data[unsafe_offset=i * stride] - wmean)
            W += wi
            wabsdev += (delta - wabsdev) * (wi / W)
    return wabsdev


def stats_wabsdev[
    w_origin: Origin, x_origin: Origin
](
    w: Pointer[Float64, w_origin],
    wstride: Int,
    data: Pointer[Float64, x_origin],
    stride: Int,
    n: Int,
) -> Float64:
    var wmean = stats_wmean(w, wstride, data, stride, n)
    return stats_wabsdev_m(w, wstride, data, stride, n, wmean)


def stats_wskew_m_sd[
    w_origin: Origin, x_origin: Origin
](
    w: Pointer[Float64, w_origin],
    wstride: Int,
    data: Pointer[Float64, x_origin],
    stride: Int,
    n: Int,
    wmean: Float64,
    wsd: Float64,
) -> Float64:
    """Weighted skewness."""
    if wsd == 0.0:
        return 0.0

    var wskew: Float64 = 0.0
    var W: Float64 = 0.0
    for i in range(n):
        var wi = w[unsafe_offset=i * wstride]
        if wi > 0.0:
            var x = (data[unsafe_offset=i * stride] - wmean) / wsd
            W += wi
            wskew += (x * x * x - wskew) * (wi / W)
    return wskew


def stats_wskew[
    w_origin: Origin, x_origin: Origin
](
    w: Pointer[Float64, w_origin],
    wstride: Int,
    data: Pointer[Float64, x_origin],
    stride: Int,
    n: Int,
) -> Float64:
    var wmean = stats_wmean(w, wstride, data, stride, n)
    var wsd = stats_wsd_m(w, wstride, data, stride, n, wmean)
    return stats_wskew_m_sd(w, wstride, data, stride, n, wmean, wsd)


def stats_wkurtosis_m_sd[
    w_origin: Origin, x_origin: Origin
](
    w: Pointer[Float64, w_origin],
    wstride: Int,
    data: Pointer[Float64, x_origin],
    stride: Int,
    n: Int,
    wmean: Float64,
    wsd: Float64,
) -> Float64:
    """Weighted excess kurtosis."""
    if wsd == 0.0:
        return 0.0

    var wavg: Float64 = 0.0
    var W: Float64 = 0.0
    for i in range(n):
        var wi = w[unsafe_offset=i * wstride]
        if wi > 0.0:
            var x = (data[unsafe_offset=i * stride] - wmean) / wsd
            W += wi
            wavg += (x * x * x * x - wavg) * (wi / W)
    return wavg - 3.0


def stats_wkurtosis[
    w_origin: Origin, x_origin: Origin
](
    w: Pointer[Float64, w_origin],
    wstride: Int,
    data: Pointer[Float64, x_origin],
    stride: Int,
    n: Int,
) -> Float64:
    var wmean = stats_wmean(w, wstride, data, stride, n)
    var wsd = stats_wsd_m(w, wstride, data, stride, n, wmean)
    return stats_wkurtosis_m_sd(w, wstride, data, stride, n, wmean, wsd)
