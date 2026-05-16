# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Shivasankar K.A.
"""
MSL (Mojo Scientific Library) — public API.

Re-exports the full public surface of all MSL submodules.
"""

comptime MutExt = MutExternalOrigin
"""Alias for mutable external origin, used for pointers to external data."""

comptime f64: DType = DType.float64
"""Alias for the 64-bit floating-point data type."""

comptime __version__: String = "0.1.0"

# Core
from msl.core import Block, block_alloc, block_calloc, block_size, block_data
from msl.core.errno import (
    MSL_SUCCESS,
    MSL_FAILURE,
    MSL_CONTINUE,
    MSL_EDOM,
    MSL_ERANGE,
    MSL_EINVAL,
    MSL_EFAILED,
    MSL_EMAXITER,
    MSL_EZERODIV,
    MSL_EBADTOL,
    MSL_ETOL,
    MSL_EUNDRFLW,
    MSL_EOVRFLW,
    MSL_ELOSS,
    MSL_EROUND,
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
    bessel_Jn,
    bessel_Yn,
    bessel_In,
    bessel_Kn,
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
    psi,
    psi_n,
    gamma_inc,
    gamma_inc_P,
    gamma_inc_Q,
    beta_inc,
)

# Integration
from msl.integration import (
    QKResult,
    IntegrationResult,
    IntegrationWorkspace,
    qk15,
    qk21,
    qk31,
    qk41,
    qk51,
    qk61,
    qng_integrate,
    qag,
    qags,
)

# RNG
from msl.rng import RNGAlgorithm, MT19937, RNG

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
    tdist,
    tdist_pdf,
    lognormal,
    lognormal_pdf,
    weibull,
    weibull_pdf,
    binomial,
    binomial_pdf,
    negative_binomial,
    negative_binomial_pdf,
    cauchy,
    cauchy_pdf,
    laplace,
    laplace_pdf,
)

# Optimization (root-finding + minimization)
from msl.optimizer import (
    RootResult,
    MinResult,
    root_bisect,
    root_brent,
    root_newton,
    root_secant,
    min_brent,
    min_golden,
)

# Interpolation
from msl.interpolation import (
    InterpResult,
    LinearInterp,
    CubicSpline,
    AkimaSpline,
)

# ODE solvers
from msl.ode import OdeResult, ode_rk4, ode_rkf45

# Numerical differentiation
from msl.deriv import DerivResult, deriv_central, deriv_forward, deriv_backward

# Linear algebra primitives
from msl.vector import (
    Vector,
    vector_alloc,
    vector_calloc,
    vector_size,
    vector_stride,
    vector_set_zero,
    vector_set_all,
    vector_add,
    vector_sub,
    vector_scale,
    vector_axpy,
    vector_dot,
    vector_norm,
)
from msl.matrix import (
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
from msl.permutation import (
    Permutation,
    permutation_alloc,
    permutation_init,
    permutation_next,
)
