comptime MutExt = MutExternalOrigin
"""Alias for mutable external origin, used for pointers to external data."""

comptime f64: DType = DType.float64
"""Alias for the 64-bit floating-point data type."""

comptime __version__: String = "0.1.0"

# Core
from msl.core import Block, block_alloc, block_calloc, block_size, block_data
from msl.core.errno import (
    GSL_SUCCESS,
    GSL_FAILURE,
    GSL_CONTINUE,
    GSL_EDOM,
    GSL_ERANGE,
    GSL_EINVAL,
    GSL_EFAILED,
    GSL_EMAXITER,
    GSL_EZERODIV,
    GSL_EBADTOL,
    GSL_ETOL,
    GSL_EUNDRFLW,
    GSL_EOVRFLW,
    GSL_ELOSS,
    GSL_EROUND,
    msl_error,
)

# Special functions
from msl.sf import (
    SFSResult,
    airy_ai,
    airy_bi,
    airy_ai_scaled,
    airy_bi_scaled,
    airy_ai_deriv,
    airy_bi_deriv,
    airy_ai_deriv_scaled,
    airy_bi_deriv_scaled,
    bessel_j0,
    bessel_j1,
    bessel_y0,
    bessel_y1,
    bessel_i0_scaled,
    bessel_i1_scaled,
    bessel_k0_scaled,
    bessel_k1_scaled,
    gamma,
    lngamma,
    gammastar,
    gammainv,
    factorial,
    double_factorial,
    ln_factorial,
    ln_double_factorial,
    beta,
    lnbeta,
    erf,
    erfc,
    log_erfc,
    erf_Z,
    erf_Q,
    hazard,
    legendre_P1,
    legendre_P2,
    legendre_P3,
    legendre_Pl,
)

# Integration
from msl.integration import (
    QKResult,
    IntegrationResult,
    qk15,
    qk21,
    qk31,
    qk41,
    qk51,
    qk61,
    qng_integrate,
)

# RNG
from msl.rng import RNG

# Distributions
from msl.distributions import (
    gaussian,
    gaussian_pdf,
    uniform,
    uniform_pdf,
    exponential,
    exponential_pdf,
    gamma as gamma_dist,
    gamma_pdf,
    beta as beta_dist,
    chisq,
    poisson,
    poisson_pdf,
)

# Linear algebra primitives
from msl.vector import Vector, vector_alloc, vector_calloc, vector_size, vector_stride, vector_set_zero, vector_set_all
from msl.matrix import Matrix, matrix_alloc, matrix_calloc, matrix_size1, matrix_size2, matrix_set_zero, matrix_set_all, matrix_set_identity
from msl.permutation import Permutation, permutation_alloc, permutation_init, permutation_next
