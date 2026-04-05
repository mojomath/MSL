# SPDX-License-Identifier: GPL-3.0-or-later

# ===----------------------------------------------------------------------=== #
# MSL (Mojo Scientific Library)
#
# Derived from GNU Scientific Library (GSL)
# Original file: integration/gsl_integration.h
#
# Original authors:
# Copyright (C) 1996-2007 Brian Gough
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
Workspace and result types for numerical integration.

This is a direct port of the GSL workspace structures.
"""

from std.memory import UnsafePointer, memset_zero

comptime MutExt = MutExternalOrigin

comptime GSL_KRONROD_15: Int = 1
comptime GSL_KRONROD_21: Int = 2
comptime GSL_KRONROD_31: Int = 3
comptime GSL_KRONROD_41: Int = 4
comptime GSL_KRONROD_51: Int = 5
comptime GSL_KRONROD_61: Int = 6


struct QKResult(Movable):
    var result: Float64
    var abserr: Float64
    var resabs: Float64
    var resasc: Float64

    def __init__(out self):
        self.result = 0.0
        self.abserr = 0.0
        self.resabs = 0.0
        self.resasc = 0.0


struct IntegrationResult(Movable):
    """Result of an integration operation."""

    var val: Float64
    var err: Float64

    def __init__(out self):
        self.val = 0.0
        self.err = 0.0
