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
Dense linear algebra: LU, Cholesky, QR decompositions and a symmetric
eigenvalue solver.
"""

# ===----------------------------------------------------------------------=== #
# MSL
# ===----------------------------------------------------------------------=== #
from msl.linalg.cholesky import (
    cholesky_decomp,
    cholesky_solve,
    cholesky_svx,
)
from msl.linalg.jacobi import eigen_jacobi
from msl.linalg.lu import (
    lu_decomp,
    lu_det,
    lu_invert,
    lu_lndet,
    lu_solve,
    lu_svx,
)
from msl.linalg.qr import (
    qr_decomp,
    qr_solve,
    qr_svx,
)
