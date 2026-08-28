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
Numerical integration routines.
"""

# ===----------------------------------------------------------------------=== #
# MSL
# ===----------------------------------------------------------------------=== #
from msl.integration.qag import (
    MSL_INTEG_GAUSS15,
    MSL_INTEG_GAUSS21,
    MSL_INTEG_GAUSS31,
    MSL_INTEG_GAUSS41,
    MSL_INTEG_GAUSS51,
    MSL_INTEG_GAUSS61,
    qag,
)
from msl.integration.qags import qags
from msl.integration.qk15 import qk15
from msl.integration.qk21 import qk21
from msl.integration.qk31 import qk31
from msl.integration.qk41 import qk41
from msl.integration.qk51 import qk51
from msl.integration.qk61 import qk61
from msl.integration.qng import qng_integrate
from msl.integration.workspace import (
    IntegrationResult,
    IntegrationWorkspace,
    MSL_KRONROD_15,
    MSL_KRONROD_21,
    MSL_KRONROD_31,
    MSL_KRONROD_41,
    MSL_KRONROD_51,
    MSL_KRONROD_61,
    QKResult,
)
