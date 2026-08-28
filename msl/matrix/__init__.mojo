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
Dense 2D matrix and linear algebra operations.
"""

from .matrix import (
    Matrix,
    matrix_alloc,
    matrix_calloc,
    matrix_size1,
    matrix_size2,
    matrix_set_zero,
    matrix_set_all,
    matrix_set_identity,
    matrix_add,
    matrix_sub,
    matrix_scale,
    matrix_transpose,
    matrix_mul,
)
