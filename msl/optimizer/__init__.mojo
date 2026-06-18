# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Shivasankar K.A.
"""
Scalar root-finding and minimization algorithms.
"""

from .utility import (
    RootResult,
    MinResult,
    MinBracketResult,
    root_test_interval,
    root_test_residual,
    root_test_delta,
    min_test_interval,
    min_find_bracket,
)
from .root_scalar import (
    root_bisect,
    root_brent,
    root_newton,
    root_secant,
    root_falsepos,
    root_steffenson,
)
from .min_scalar import min_brent, min_golden
from .root_iter import RootFSolver, RootFDFSolver
from .min_iter import MinFSolver
