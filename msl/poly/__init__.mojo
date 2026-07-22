# SPDX-License-Identifier: GPL-3.0-or-later

"""
Polynomial evaluation, interpolation, and real-root solving.
"""

from .poly import (
    poly_eval,
    poly_eval_derivs,
    poly_dd_init,
    poly_dd_eval,
    poly_dd_taylor,
)

from .solve import (
    QuadraticRoots,
    CubicRoots,
    poly_solve_quadratic,
    poly_solve_cubic,
)
