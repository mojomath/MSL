# SPDX-License-Identifier: GPL-3.0-or-later

# ===----------------------------------------------------------------------=== #
# MSL (Mojo Scientific Library)
#
# Derived from GNU Scientific Library (GSL)
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

"""Special functions (sf) module.

This module implements special mathematical functions from GSL.
"""

from .airy import (
    airy_ai,
    airy_bi,
    airy_ai_scaled,
    airy_bi_scaled,
    airy_ai_deriv,
    airy_bi_deriv,
    airy_ai_deriv_scaled,
    airy_bi_deriv_scaled,
)

from .bessel import (
    bessel_j0,
    bessel_j1,
    bessel_y0,
    bessel_y1,
    bessel_i0_scaled,
    bessel_i1_scaled,
    bessel_k0_scaled,
    bessel_k1_scaled,
)

from .gamma import (
    gamma,
    lngamma,
    gammastar,
    gammainv,
    factorial,
    double_factorial,
    ln_factorial,
    ln_double_factorial,
)

from .beta import (
    beta,
    lnbeta,
)

from .erf import (
    erf,
    erfc,
    log_erfc,
    erf_Z,
    erf_Q,
    hazard,
)

from .legendre import (
    legendre_P1,
    legendre_P2,
    legendre_P3,
    legendre_Pl,
)

from .result import SFSResult

comptime PrecisionDouble: Int = 0
comptime PrecisionSingle: Int = 1
comptime PrecisionApprox: Int = 2
