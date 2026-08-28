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
Scalar root-finding and minimization algorithms.
"""

# ===----------------------------------------------------------------------=== #
# MSL
# ===----------------------------------------------------------------------=== #
from msl.optimizer.min_iter import MinFSolver
from msl.optimizer.min_scalar import (
    min_brent,
    min_golden,
)
from msl.optimizer.root_iter import (
    RootFDFSolver,
    RootFSolver,
)
from msl.optimizer.root_scalar import (
    root_bisect,
    root_brent,
    root_falsepos,
    root_newton,
    root_secant,
    root_steffenson,
)
from msl.optimizer.utility import (
    min_find_bracket,
    min_test_interval,
    MinBracketResult,
    MinResult,
    root_test_delta,
    root_test_interval,
    root_test_residual,
    RootResult,
)
