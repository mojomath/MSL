# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Shivasankar K.A.
"""
Scalar root-finding and minimization algorithms.
"""

from .utility import RootResult, MinResult
from .root_scalar import root_bisect, root_brent, root_newton, root_secant
from .min_scalar import min_brent, min_golden
