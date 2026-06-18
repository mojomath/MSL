# SPDX-License-Identifier: GPL-3.0-or-later

from std.testing import TestSuite
from std.math import abs, sqrt

from msl.statistics import (
    stats_spearman,
    stats_mad0,
    stats_mad,
    stats_Sn0_from_sorted_data,
    stats_Sn_from_sorted_data,
    stats_Qn0_from_sorted_data,
    stats_Qn_from_sorted_data,
    stats_max,
    stats_min,
    stats_minmax,
    stats_max_index,
    stats_min_index,
    stats_minmax_index,
    stats_select,
    stats_median_from_sorted_data,
    stats_median,
    stats_quantile_from_sorted_data,
    stats_trmean_from_sorted_data,
    stats_gastwirth_from_sorted_data,
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
    stats_pvariance,
    stats_ttest,
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


def test_spearman() raises:
    # GSL rawa / rawb (n=14 each) -> spearman = -0.1604395604395604396
    comptime n = 14
    var a = alloc[Float64](n)
    var b = alloc[Float64](n)
    a[0]  = 0.0421
    a[1]  = 0.0941
    a[2]  = 0.1064
    a[3]  = 0.0242
    a[4]  = 0.1331
    a[5]  = 0.0773
    a[6]  = 0.0243
    a[7]  = 0.0815
    a[8]  = 0.1186
    a[9]  = 0.0356
    a[10] = 0.0728
    a[11] = 0.0999
    a[12] = 0.0614
    a[13] = 0.0479
    b[0]  = 0.1081
    b[1]  = 0.0986
    b[2]  = 0.1566
    b[3]  = 0.1961
    b[4]  = 0.1125
    b[5]  = 0.1942
    b[6]  = 0.1079
    b[7]  = 0.1021
    b[8]  = 0.1583
    b[9]  = 0.1673
    b[10] = 0.1675
    b[11] = 0.1856
    b[12] = 0.1688
    b[13] = 0.1512

    var r1 = alloc[Float64](n)
    var r2 = alloc[Float64](n)
    assert tol(stats_spearman(a, 1, b, 1, n, r1, r2), -0.1604395604395604396, 1e-10)

    # perfectly increasing -> rho = 1
    var x = alloc[Float64](5)
    var y = alloc[Float64](5)
    var wx = alloc[Float64](5)
    var wy = alloc[Float64](5)
    for i in range(5):
        x[i] = Float64(i)
        y[i] = Float64(i)
    assert tol(stats_spearman(x, 1, y, 1, 5, wx, wy), 1.0, 1e-12)

    # perfectly decreasing -> rho = -1
    for i in range(5):
        y[i] = Float64(4 - i)
    assert tol(stats_spearman(x, 1, y, 1, 5, wx, wy), -1.0, 1e-12)

    wy.free()
    wx.free()
    y.free()
    x.free()
    r2.free()
    r1.free()
    b.free()
    a.free()
    print("test_spearman: PASSED")


def test_mad() raises:
    # GSL rawa unsorted, n=14 -> mad0=0.02925, n=13 -> mad0=0.02910
    comptime n = 14
    var data = alloc[Float64](n)
    data[0]  = 0.0421
    data[1]  = 0.0941
    data[2]  = 0.1064
    data[3]  = 0.0242
    data[4]  = 0.1331
    data[5]  = 0.0773
    data[6]  = 0.0243
    data[7]  = 0.0815
    data[8]  = 0.1186
    data[9]  = 0.0356
    data[10] = 0.0728
    data[11] = 0.0999
    data[12] = 0.0614
    data[13] = 0.0479

    var work = alloc[Float64](n)
    assert tol(stats_mad0(data, 1, n, work), 0.02925, 1e-4)
    assert tol(stats_mad0(data, 1, n - 1, work), 0.02910, 1e-4)
    assert tol(stats_mad(data, 1, n, work), 1.482602218505602 * 0.02925, 1e-4)

    work.free()
    data.free()
    print("test_mad: PASSED")


def test_Sn() raises:
    # GSL rawa sorted, n=14 -> Sn=0.04007136, n=13 -> Sn=0.03728599834710744
    comptime n = 14
    var sorted = alloc[Float64](n)
    sorted[0]  = 0.0242
    sorted[1]  = 0.0243
    sorted[2]  = 0.0356
    sorted[3]  = 0.0421
    sorted[4]  = 0.0479
    sorted[5]  = 0.0614
    sorted[6]  = 0.0728
    sorted[7]  = 0.0773
    sorted[8]  = 0.0815
    sorted[9]  = 0.0941
    sorted[10] = 0.0999
    sorted[11] = 0.1064
    sorted[12] = 0.1186
    sorted[13] = 0.1331

    var work = alloc[Float64](n)
    assert tol(stats_Sn_from_sorted_data(sorted, 1, n, work), 0.04007136, 1e-6)
    assert tol(stats_Sn_from_sorted_data(sorted, 1, n - 1, work), 0.03728599834710744, 1e-8)

    work.free()
    sorted.free()
    print("test_Sn: PASSED")


def test_Qn() raises:
    # GSL rawa sorted, n=14 -> Qn=0.04113672759664409, n=13 -> Qn=0.03684305546303433
    comptime n = 14
    var sorted = alloc[Float64](n)
    sorted[0]  = 0.0242
    sorted[1]  = 0.0243
    sorted[2]  = 0.0356
    sorted[3]  = 0.0421
    sorted[4]  = 0.0479
    sorted[5]  = 0.0614
    sorted[6]  = 0.0728
    sorted[7]  = 0.0773
    sorted[8]  = 0.0815
    sorted[9]  = 0.0941
    sorted[10] = 0.0999
    sorted[11] = 0.1064
    sorted[12] = 0.1186
    sorted[13] = 0.1331

    var work = alloc[Float64](3 * n)
    var work_int = alloc[Int](5 * n)
    assert tol(stats_Qn_from_sorted_data(sorted, 1, n, work, work_int), 0.04113672759664409, 1e-8)
    assert tol(stats_Qn_from_sorted_data(sorted, 1, n - 1, work, work_int), 0.03684305546303433, 1e-8)

    work_int.free()
    work.free()
    sorted.free()
    print("test_Qn: PASSED")


def test_minmax_and_indices() raises:
    # GSL rawa: max=0.1331 at index 4, min=0.0242 at index 3
    comptime n = 14
    var data = alloc[Float64](n)
    data[0]  = 0.0421
    data[1]  = 0.0941
    data[2]  = 0.1064
    data[3]  = 0.0242
    data[4]  = 0.1331
    data[5]  = 0.0773
    data[6]  = 0.0243
    data[7]  = 0.0815
    data[8]  = 0.1186
    data[9]  = 0.0356
    data[10] = 0.0728
    data[11] = 0.0999
    data[12] = 0.0614
    data[13] = 0.0479

    assert stats_max(data, 1, n) == 0.1331
    assert stats_min(data, 1, n) == 0.0242
    assert stats_max_index(data, 1, n) == 4
    assert stats_min_index(data, 1, n) == 3

    var mm = stats_minmax(data, 1, n)
    assert mm[0] == 0.0242
    assert mm[1] == 0.1331

    var mmi = stats_minmax_index(data, 1, n)
    assert mmi[0] == 3
    assert mmi[1] == 4

    data.free()
    print("test_minmax_and_indices: PASSED")


def test_median_and_select() raises:
    # GSL rawa unsorted, n=14 -> even median = 0.07505, n=13 -> odd median = 0.0728
    comptime n = 14
    var data = alloc[Float64](n)
    data[0]  = 0.0421
    data[1]  = 0.0941
    data[2]  = 0.1064
    data[3]  = 0.0242
    data[4]  = 0.1331
    data[5]  = 0.0773
    data[6]  = 0.0243
    data[7]  = 0.0815
    data[8]  = 0.1186
    data[9]  = 0.0356
    data[10] = 0.0728
    data[11] = 0.0999
    data[12] = 0.0614
    data[13] = 0.0479

    assert tol(stats_median(data, 1, n), 0.07505, 1e-10)

    # reload for odd case (median mutates)
    data[0]  = 0.0421
    data[1]  = 0.0941
    data[2]  = 0.1064
    data[3]  = 0.0242
    data[4]  = 0.1331
    data[5]  = 0.0773
    data[6]  = 0.0243
    data[7]  = 0.0815
    data[8]  = 0.1186
    data[9]  = 0.0356
    data[10] = 0.0728
    data[11] = 0.0999
    data[12] = 0.0614

    assert tol(stats_median(data, 1, n - 1), 0.0728, 1e-10)

    data.free()
    print("test_median_and_select: PASSED")


def test_median_from_sorted_data() raises:
    # GSL rawa sorted (ascending)
    comptime n = 14
    var sorted = alloc[Float64](n)
    sorted[0]  = 0.0242
    sorted[1]  = 0.0243
    sorted[2]  = 0.0356
    sorted[3]  = 0.0421
    sorted[4]  = 0.0479
    sorted[5]  = 0.0614
    sorted[6]  = 0.0728
    sorted[7]  = 0.0773
    sorted[8]  = 0.0815
    sorted[9]  = 0.0941
    sorted[10] = 0.0999
    sorted[11] = 0.1064
    sorted[12] = 0.1186
    sorted[13] = 0.1331

    # even: (sorted[6] + sorted[7]) / 2 = (0.0728 + 0.0773) / 2 = 0.07505
    assert tol(stats_median_from_sorted_data(sorted, 1, n), 0.07505, 1e-10)
    # odd: sorted[6] = 0.0728
    assert tol(stats_median_from_sorted_data(sorted, 1, n - 1), 0.0728, 1e-10)

    sorted.free()
    print("test_median_from_sorted_data: PASSED")


def test_quantile_from_sorted_data() raises:
    comptime n = 14
    var sorted = alloc[Float64](n)
    sorted[0]  = 0.0242
    sorted[1]  = 0.0243
    sorted[2]  = 0.0356
    sorted[3]  = 0.0421
    sorted[4]  = 0.0479
    sorted[5]  = 0.0614
    sorted[6]  = 0.0728
    sorted[7]  = 0.0773
    sorted[8]  = 0.0815
    sorted[9]  = 0.0941
    sorted[10] = 0.0999
    sorted[11] = 0.1064
    sorted[12] = 0.1186
    sorted[13] = 0.1331

    assert tol(stats_quantile_from_sorted_data(sorted, 1, n, 0.0), 0.0242, 1e-10)
    assert tol(stats_quantile_from_sorted_data(sorted, 1, n, 1.0), 0.1331, 1e-10)
    assert tol(stats_quantile_from_sorted_data(sorted, 1, n, 0.5), 0.07505, 1e-10)

    sorted.free()
    print("test_quantile_from_sorted_data: PASSED")


def test_trmean_and_gastwirth() raises:
    comptime n = 14
    var sorted = alloc[Float64](n)
    sorted[0]  = 0.0242
    sorted[1]  = 0.0243
    sorted[2]  = 0.0356
    sorted[3]  = 0.0421
    sorted[4]  = 0.0479
    sorted[5]  = 0.0614
    sorted[6]  = 0.0728
    sorted[7]  = 0.0773
    sorted[8]  = 0.0815
    sorted[9]  = 0.0941
    sorted[10] = 0.0999
    sorted[11] = 0.1064
    sorted[12] = 0.1186
    sorted[13] = 0.1331

    # trim=0.2, even n=14
    assert tol(stats_trmean_from_sorted_data(0.2, sorted, 1, n), 0.0719, 1e-4)
    # trim=0.2, odd n=13
    assert tol(stats_trmean_from_sorted_data(0.2, sorted, 1, n - 1), 0.06806666666666666, 1e-10)

    # gastwirth even n=14
    assert tol(stats_gastwirth_from_sorted_data(sorted, 1, n), 0.07271, 1e-4)
    # gastwirth odd n=13
    assert tol(stats_gastwirth_from_sorted_data(sorted, 1, n - 1), 0.06794, 1e-4)

    sorted.free()
    print("test_trmean_and_gastwirth: PASSED")


def test_pvariance_and_ttest() raises:
    # GSL test datasets (test_float_source.c rawa / rawb, n=14 each)
    comptime na = 14
    comptime nb = 14
    var a = alloc[Float64](na)
    var b = alloc[Float64](nb)
    a[0]  = 0.0421
    a[1]  = 0.0941
    a[2]  = 0.1064
    a[3]  = 0.0242
    a[4]  = 0.1331
    a[5]  = 0.0773
    a[6]  = 0.0243
    a[7]  = 0.0815
    a[8]  = 0.1186
    a[9]  = 0.0356
    a[10] = 0.0728
    a[11] = 0.0999
    a[12] = 0.0614
    a[13] = 0.0479
    b[0]  = 0.1081
    b[1]  = 0.0986
    b[2]  = 0.1566
    b[3]  = 0.1961
    b[4]  = 0.1125
    b[5]  = 0.1942
    b[6]  = 0.1079
    b[7]  = 0.1021
    b[8]  = 0.1583
    b[9]  = 0.1673
    b[10] = 0.1675
    b[11] = 0.1856
    b[12] = 0.1688
    b[13] = 0.1512

    assert tol(stats_pvariance(a, 1, na, b, 1, nb), 0.00123775384615385, 1e-10)
    assert tol(stats_ttest(a, 1, na, b, 1, nb), -5.67026326985851, 1e-8)

    a.free()
    b.free()
    print("test_pvariance_and_ttest: PASSED")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("All statistics tests PASSED")
