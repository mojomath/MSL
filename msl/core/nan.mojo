# SPDX-License-Identifier: GPL-3.0-or-later

# ===----------------------------------------------------------------------=== #
# MSL (Mojo Scientific Library)
#
# Derived from GNU Scientific Library (GSL)
# Original file: gsl_nan.h
#
# Original authors:
# Copyright (C) 1996–2007 Gerard Jungman, Brian Gough
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
NaN, Infinity, and related values.

Special floating point values.
"""

from std.utils.numerics import nan, isnan, isinf, isfinite


# ===----------------------------------------------------------------------=== #
# Special values
# ===----------------------------------------------------------------------=== #

comptime MSL_POSINF: Float64 = Float64.MAX
"""Positive infinity."""

comptime MSL_NEGINF: Float64 = -Float64.MAX
"""Negative infinity."""

comptime MSL_NAN: Float64 = nan[DType.float64]()
"""Not-a-number."""

comptime MSL_POSZERO: Float64 = 0.0
"""Positive zero."""

# NOTE: is this really the correct way to define this in mojo?
comptime MSL_NEGZERO: Float64 = -0.0
"""Negative zero."""


# ===----------------------------------------------------------------------=== #
# Predicates
# ===----------------------------------------------------------------------=== #


def msl_isnan(x: Float64) -> Bool:
    """Return True if x is NaN."""
    return isnan(x)


def msl_isinf(x: Float64) -> Int:
    """Return +1 if x is +infinity, -1 if x is -infinity, 0 otherwise."""
    if isinf(x):
        return 1 if x > 0.0 else -1
    else:
        return 0


def msl_isfinite(x: Float64) -> Bool:
    """Return True if x is a finite number."""
    return isfinite(x)


def msl_posinf() -> Float64:
    """Return positive infinity."""
    return MSL_POSINF


def msl_neginf() -> Float64:
    """Return negative infinity."""
    return MSL_NEGINF


def msl_nan() -> Float64:
    """Return NaN."""
    return MSL_NAN
