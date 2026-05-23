# SPDX-License-Identifier: GPL-3.0-or-later

from std.testing import TestSuite
from std.math import abs, sqrt

from msl.statistics import (
    stats_mean,
    stats_variance,
    stats_sd,
    stats_variance_with_fixed_mean,
    stats_sd_with_fixed_mean,
    stats_tss,
    stats_tss_m,
    stats_absdev,
    stats_absdev_m,
    stats_skew,
    stats_kurtosis,
    stats_lag1_autocorrelation,
    stats_covariance,
    stats_covariance_m,
    stats_correlation,
    stats_variance_m,
    stats_sd_m,
    stats_wmean,
    stats_wvariance,
    stats_wsd,
    stats_wvariance_with_fixed_mean,
    stats_wsd_with_fixed_mean,
    stats_wtss,
    stats_wtss_m,
    stats_wabsdev,
    stats_wabsdev_m,
    stats_wskew,
    stats_wkurtosis,
    stats_wvariance_m,
    stats_wsd_m,
)


def tol(x: Float64, expected: Float64, t: Float64) -> Bool:
    return abs(x - expected) < t


def test_moments_basic() raises:
    var data = alloc[Float64](5)
    data[0] = 1.0
    data[1] = 2.0
    data[2] = 3.0
    data[3] = 4.0
    data[4] = 5.0

    assert tol(stats_mean(data, 1, 5), 3.0, 1e-12)
    assert tol(stats_variance(data, 1, 5), 2.5, 1e-12)
    assert tol(stats_sd(data, 1, 5), sqrt(2.5), 1e-12)
    assert tol(stats_tss(data, 1, 5), 10.0, 1e-12)
    assert tol(stats_absdev(data, 1, 5), 1.2, 1e-12)

    data.free()
    print("test_moments_basic: PASSED")


def test_fixed_mean_variants() raises:
    var data = alloc[Float64](5)
    data[0] = 1.0
    data[1] = 2.0
    data[2] = 3.0
    data[3] = 4.0
    data[4] = 5.0

    var mean = 3.0
    assert tol(stats_variance_with_fixed_mean(data, 1, 5, mean), 2.0, 1e-12)
    assert tol(stats_sd_with_fixed_mean(data, 1, 5, mean), sqrt(2.0), 1e-12)
    assert tol(stats_variance_m(data, 1, 5, mean), 2.5, 1e-12)
    assert tol(stats_sd_m(data, 1, 5, mean), sqrt(2.5), 1e-12)
    assert tol(stats_tss_m(data, 1, 5, mean), 10.0, 1e-12)
    assert tol(stats_absdev_m(data, 1, 5, mean), 1.2, 1e-12)

    data.free()
    print("test_fixed_mean_variants: PASSED")


def test_skew_kurtosis_symmetric_data() raises:
    var data = alloc[Float64](5)
    data[0] = 1.0
    data[1] = 2.0
    data[2] = 3.0
    data[3] = 4.0
    data[4] = 5.0

    assert tol(stats_skew(data, 1, 5), 0.0, 1e-12)
    assert tol(stats_kurtosis(data, 1, 5), -1.912, 1e-3)

    data.free()
    print("test_skew_kurtosis_symmetric_data: PASSED")


def test_lag1_autocorrelation() raises:
    var data = alloc[Float64](4)
    data[0] = 1.0
    data[1] = -1.0
    data[2] = 1.0
    data[3] = -1.0

    assert tol(stats_lag1_autocorrelation(data, 1, 4), -0.75, 1e-12)

    data.free()
    print("test_lag1_autocorrelation: PASSED")


def test_covariance_and_correlation() raises:
    var x = alloc[Float64](5)
    var y = alloc[Float64](5)
    x[0] = 1.0
    x[1] = 2.0
    x[2] = 3.0
    x[3] = 4.0
    x[4] = 5.0
    y[0] = 3.0
    y[1] = 5.0
    y[2] = 7.0
    y[3] = 9.0
    y[4] = 11.0

    assert tol(stats_covariance(x, 1, y, 1, 5), 5.0, 1e-12)
    assert tol(stats_covariance_m(x, 1, y, 1, 5, 3.0, 7.0), 5.0, 1e-12)
    assert tol(stats_correlation(x, 1, y, 1, 5), 1.0, 1e-12)

    x.free()
    y.free()
    print("test_covariance_and_correlation: PASSED")


def test_stride_behavior() raises:
    var data = alloc[Float64](10)
    data[0] = 1.0
    data[1] = 99.0
    data[2] = 2.0
    data[3] = 99.0
    data[4] = 3.0
    data[5] = 99.0
    data[6] = 4.0
    data[7] = 99.0
    data[8] = 5.0
    data[9] = 99.0

    assert tol(stats_mean(data, 2, 5), 3.0, 1e-12)
    assert tol(stats_variance(data, 2, 5), 2.5, 1e-12)

    data.free()
    print("test_stride_behavior: PASSED")


def test_weighted_equal_weights_reduce_to_unweighted() raises:
    var data = alloc[Float64](5)
    var w = alloc[Float64](5)
    for i in range(5):
        data[i] = Float64(i + 1)
        w[i] = 1.0

    assert tol(stats_wmean(w, 1, data, 1, 5), 3.0, 1e-12)
    assert tol(stats_wvariance(w, 1, data, 1, 5), 2.5, 1e-12)
    assert tol(stats_wsd(w, 1, data, 1, 5), sqrt(2.5), 1e-12)
    assert tol(stats_wtss(w, 1, data, 1, 5), 10.0, 1e-12)
    assert tol(stats_wabsdev(w, 1, data, 1, 5), 1.2, 1e-12)

    w.free()
    data.free()
    print("test_weighted_equal_weights_reduce_to_unweighted: PASSED")


def test_weighted_fixed_mean_variants() raises:
    var data = alloc[Float64](4)
    var w = alloc[Float64](4)
    data[0] = 1.0
    data[1] = 2.0
    data[2] = 3.0
    data[3] = 4.0
    w[0] = 1.0
    w[1] = 2.0
    w[2] = 3.0
    w[3] = 4.0

    var wm = stats_wmean(w, 1, data, 1, 4)  # 3.0
    assert tol(wm, 3.0, 1e-12)
    assert tol(stats_wvariance_with_fixed_mean(w, 1, data, 1, 4, wm), 1.0, 1e-12)
    assert tol(stats_wsd_with_fixed_mean(w, 1, data, 1, 4, wm), 1.0, 1e-12)
    assert tol(stats_wvariance_m(w, 1, data, 1, 4, wm), 1.4285714285714286, 1e-12)
    assert tol(stats_wsd_m(w, 1, data, 1, 4, wm), 1.1952286093343936, 1e-12)
    assert tol(stats_wtss_m(w, 1, data, 1, 4, wm), 10.0, 1e-12)
    assert tol(stats_wabsdev_m(w, 1, data, 1, 4, wm), 0.8, 1e-12)

    w.free()
    data.free()
    print("test_weighted_fixed_mean_variants: PASSED")


def test_weighted_skew_kurtosis_symmetric() raises:
    var data = alloc[Float64](5)
    var w = alloc[Float64](5)
    data[0] = 1.0
    data[1] = 2.0
    data[2] = 3.0
    data[3] = 4.0
    data[4] = 5.0
    w[0] = 1.0
    w[1] = 1.0
    w[2] = 1.0
    w[3] = 1.0
    w[4] = 1.0

    assert tol(stats_wskew(w, 1, data, 1, 5), 0.0, 1e-12)
    assert tol(stats_wkurtosis(w, 1, data, 1, 5), -1.912, 1e-3)

    w.free()
    data.free()
    print("test_weighted_skew_kurtosis_symmetric: PASSED")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("All statistics tests PASSED")
