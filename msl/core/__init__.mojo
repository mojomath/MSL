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
Core types, error codes, and math utilities.
"""

# ===----------------------------------------------------------------------=== #
# MSL
# ===----------------------------------------------------------------------=== #
from msl.core.block import (
    Block,
    block_alloc,
    block_calloc,
    block_data,
    block_size,
)
from msl.core.errno import (
    MSL_EINVAL,
    MSL_ENOMEM,
    MSL_FAILURE,
    MSL_SUCCESS,
)
from msl.core.minmax import (
    max,
    min,
)
from msl.core.nan import (
    msl_isinf,
    msl_isnan,
)
from msl.core.pow_int import msl_pow_int
