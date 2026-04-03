# SPDX-License-Identifier: GPL-3.0-or-later

# ===----------------------------------------------------------------------=== #
# MSL (Mojo Scientific Library)
#
# Derived from GNU Scientific Library (GSL)
# Original file: gsl_minmax.h
#
# Original authors:
# Copyright (C) 2008 Gerard Jungman, Brian Gough
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
Min/max operations.

Utilities for finding minimum and maximum values.
"""


# ===----------------------------------------------------------------------=== #
# Basic max/min
# ===----------------------------------------------------------------------=== #

def msl_max(a: Float64, b: Float64) -> Float64:
    """Return the larger of a and b."""
    return a if a > b else b


def msl_min(a: Float64, b: Float64) -> Float64:
    """Return the smaller of a and b."""
    return a if a < b else b


def msl_max(a: Int, b: Int) -> Int:
    """Return the larger of a and b (integers)."""
    return a if a > b else b


def msl_min(a: Int, b: Int) -> Int:
    """Return the smaller of a and b (integers)."""
    return a if a < b else b
