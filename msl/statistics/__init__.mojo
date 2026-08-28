# SPDX-License-Identifier: GPL-3.0-or-later

# ===----------------------------------------------------------------------=== #
# MSL (Mojo Scientific Library)
#
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
Descriptive statistics routines.
"""

# ===----------------------------------------------------------------------=== #
# MSL
# ===----------------------------------------------------------------------=== #
from msl.statistics.moments import (
    stats_absdev,
    stats_absdev_m,
    stats_correlation,
    stats_covariance,
    stats_covariance_m,
    stats_kurtosis,
    stats_kurtosis_m_sd,
    stats_lag1_autocorrelation,
    stats_lag1_autocorrelation_m,
    stats_mean,
    stats_sd,
    stats_sd_m,
    stats_sd_with_fixed_mean,
    stats_skew,
    stats_skew_m_sd,
    stats_tss,
    stats_tss_m,
    stats_variance,
    stats_variance_m,
    stats_variance_with_fixed_mean,
)
from msl.statistics.order import (
    stats_gastwirth_from_sorted_data,
    stats_max,
    stats_max_index,
    stats_median,
    stats_median_from_sorted_data,
    stats_min,
    stats_min_index,
    stats_minmax,
    stats_minmax_index,
    stats_quantile_from_sorted_data,
    stats_select,
    stats_trmean_from_sorted_data,
)
from msl.statistics.robust import (
    stats_mad,
    stats_mad0,
    stats_Qn0_from_sorted_data,
    stats_Qn_from_sorted_data,
    stats_Sn0_from_sorted_data,
    stats_Sn_from_sorted_data,
)
from msl.statistics.spearman import stats_spearman
from msl.statistics.two_sample import (
    stats_pvariance,
    stats_ttest,
)
from msl.statistics.weighted import (
    stats_wabsdev,
    stats_wabsdev_m,
    stats_wkurtosis,
    stats_wkurtosis_m_sd,
    stats_wmean,
    stats_wsd,
    stats_wsd_m,
    stats_wsd_with_fixed_mean,
    stats_wskew,
    stats_wskew_m_sd,
    stats_wtss,
    stats_wtss_m,
    stats_wvariance,
    stats_wvariance_m,
    stats_wvariance_with_fixed_mean,
)
