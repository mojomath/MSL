# SPDX-License-Identifier: GPL-3.0-or-later

"""
Descriptive statistics routines.
"""

from .moments import (
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
    stats_skew_m_sd,
    stats_kurtosis,
    stats_kurtosis_m_sd,
    stats_lag1_autocorrelation,
    stats_lag1_autocorrelation_m,
    stats_covariance,
    stats_covariance_m,
    stats_correlation,
    stats_variance_m,
    stats_sd_m,
)

from .weighted import (
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
    stats_wskew_m_sd,
    stats_wkurtosis,
    stats_wkurtosis_m_sd,
    stats_wvariance_m,
    stats_wsd_m,
)
