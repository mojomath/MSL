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
Polynomial evaluation, interpolation, and real-root solving.
"""

# ===----------------------------------------------------------------------=== #
# MSL
# ===----------------------------------------------------------------------=== #
from msl.poly.poly import (
    poly_dd_eval,
    poly_dd_init,
    poly_dd_taylor,
    poly_eval,
    poly_eval_derivs,
)
from msl.poly.solve import (
    CubicRoots,
    poly_solve_cubic,
    poly_solve_quadratic,
    QuadraticRoots,
)
