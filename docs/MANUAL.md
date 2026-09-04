# MSL User Manual

A working guide to the library: what the functions are, what they expect,
and why a few things are spelled the way they are.

This manual is written for someone who already knows the [GNU Scientific
Library (GSL)](https://www.gnu.org/software/gsl/) and is meeting Mojo's
idioms for the first time. Most of MSL can be guessed at from GSL's own
manual - the function names, argument order, and algorithms are deliberately
close ports. What differs is how a callback is spelled, how errors are
reported, and how memory is passed around, which are recurring shapes
worth reading once, in [Passing a function to
MSL](#passing-a-function-to-msl) and [Errors](#errors), before the rest.

The per-symbol reference lives in the docstrings on every public function.
This manual is the prose half: the shape of the API and the gotchas, not an
enumeration of every signature.

- [MSL User Manual](#msl-user-manual)
  - [Getting started](#getting-started)
    - [Passing a function to MSL](#passing-a-function-to-msl)
  - [Special functions (`msl.sf`)](#special-functions-mslsf)
  - [Integration (`msl.integration`)](#integration-mslintegration)
  - [Numerical differentiation (`msl.deriv`)](#numerical-differentiation-mslderiv)
  - [Interpolation (`msl.interpolation`)](#interpolation-mslinterpolation)
  - [Root-finding and minimization (`msl.optimizer`)](#root-finding-and-minimization-msloptimizer)
  - [ODE solving (`msl.ode`)](#ode-solving-mslode)
  - [Random numbers and distributions (`msl.rng`, `msl.distributions`)](#random-numbers-and-distributions-mslrng-msldistributions)
  - [Vectors and matrices (`msl.vector`, `msl.matrix`, `msl.blas`)](#vectors-and-matrices-mslvector-mslmatrix-mslblas)
  - [Dense linear algebra (`msl.linalg`)](#dense-linear-algebra-msllinalg)
  - [Polynomials (`msl.poly`)](#polynomials-mslpoly)
  - [Permutations (`msl.permutation`)](#permutations-mslpermutation)
  - [Statistics (`msl.statistics`)](#statistics-mslstatistics)
  - [Errors](#errors)
  - [Appendix: relationship to GSL](#appendix-relationship-to-gsl)
  - [Appendix: what is not here yet](#appendix-what-is-not-here-yet)

---

## Getting started

MSL is **scalar-first**, matching GSL: every function operates on `Float64`
values (or `Int` for counts/indices), never on arrays or NDArrays directly.
Where GSL takes a `gsl_vector*`/flat `double*` buffer, MSL takes a
`Pointer[Float64, origin]` plus an explicit length (and, for matrices, a
leading dimension `lda`) - you allocate and own the memory, MSL operates on
it in place. Array-level convenience (allocate-and-return, broadcasting,
NDArray interop) is [SciJo](https://github.com/mojomath/SciJo)'s job, built
on top of MSL.

Add MSL to a project with pixi - see the [README](../README.md#installation)
for the exact `pixi.toml` stanza. Every example below assumes:

```mojo
from msl.sf import ...   # or whichever submodule
```

### Passing a function to MSL

Every routine that takes a callback (`qag`, `deriv_central`, `root_brent`,
`ode_rk4`, ...) takes it as a **compile-time parameter**, not a runtime
argument - this is what lets Mojo inline and specialize the numerics for
your specific function with no indirect-call overhead:

```mojo
def integrand(x: Float64) capturing -> Float64:
    return x * x

var r = qags[integrand](0.0, 1.0, 1e-10, 1e-10)
```

The callback goes in the `[...]` brackets, not the `(...)` call arguments.
A nested `def` defined right before the call, marked `capturing`, is the
usual shape - it can close over outer variables (constants, other
functions) the same way a Python closure would.

ODE right-hand sides take a slightly richer shape - `t`, an input `y`
pointer, and an output `dydt` pointer to write into (see [ODE
solving](#ode-solving-mslode)).

---

## Special functions (`msl.sf`)

Every special function returns an `SFSResult`:

```mojo
struct SFSResult:
    var val: Float64    # computed value
    var err: Float64     # absolute error estimate
    var errno: Int        # 0 (MSL_SUCCESS) on success, an MSL_E* code otherwise
```

**Always check `.errno`, not just `.val`**, if the input might be out of
domain - `.val` is not guaranteed to be `NaN` or otherwise obviously wrong
on failure (matching GSL's own `gsl_sf_result` convention).

```mojo
from msl.sf import airy_ai, bessel_j0, bessel_Jn, gamma, erf, legendre_Pl

var ai = airy_ai(0.0)          # Ai(0) = 0.355028...
print(ai.val, ai.err)

print(bessel_j0(1.0).val)       # J0(1) = 0.765197...
print(bessel_Jn(5, 2.0).val)     # J5(2), integer-order Bessel
print(gamma(5.0).val)             # Gamma(5) = 24
print(erf(1.0).val)                # erf(1) = 0.842700...
print(legendre_Pl(10, 0.5).val)     # Legendre P_10(0.5)
```

Covered so far: Airy (`airy_ai`/`airy_bi` and scaled/derivative variants),
Bessel J/Y (orders 0, 1, and integer `n` via `bessel_Jn`/`bessel_Yn`),
modified Bessel I/K (scaled, orders 0/1, and integer `n`), Gamma and
friends (`gamma`, `lngamma`, `gammastar`, `gammainv`, factorials), Beta,
Error functions (`erf`, `erfc`, `log_erfc`, `erf_Z`, `erf_Q`, `hazard`),
Legendre polynomials `P1`-`P3` and general `Pl`, digamma/polygamma
(`psi`/`psi_n`), and the incomplete Gamma/Beta functions.

**Gotcha - Bessel functions have distinct small-`x` and large-`x` code
paths.** `bessel_J1`/`bessel_Y1` (and the `j1`/`y1` aliases) switch to an
asymptotic Chebyshev-series amplitude/phase expansion once `|x| > 4`; the
small-`x` path uses an entirely different coefficient table. If you're
adding a test or a new Bessel-family function, exercise both ranges - a bug
in one path can be invisible from the other (this actually happened: the
large-`x` `bth1_c` coefficient table was corrupted for a while and only a
`|x| > 4` test caught it - see `docs/changelog.md` [v0.2.0]).

Not yet ported: complete spherical Bessel functions, elliptic integrals,
hypergeometric functions.

---

## Integration (`msl.integration`)

Fixed-order Gauss-Kronrod rules `qk15` through `qk61` (15- to 61-point),
`qng_integrate` (non-adaptive, tries 10 → 21 → 43 → 87 points until the
tolerance is met), and two adaptive drivers:

```mojo
from msl.integration import qags, qag, qng_integrate, MSL_INTEG_GAUSS21

def integrand(x: Float64) capturing -> Float64:
    return x * x

# Adaptive with Wynn epsilon extrapolation - good default for smooth
# integrands and integrands with endpoint singularities.
var r = qags[integrand](0.0, 1.0, 1e-10, 1e-10)
print(r.val, r.err)   # ≈ 0.333333

# General adaptive bisection, with a choice of Gauss-Kronrod rule via `key`.
var r2 = qag[integrand](0.0, 1.0, 1e-10, 1e-10, key=MSL_INTEG_GAUSS21)
print(r2.val)
```

`qag`'s `key` parameter is one of `MSL_INTEG_GAUSS{15,21,31,41,51,61}` -
higher order converges faster for smooth integrands but costs more
evaluations per subinterval. Both `qag` and `qags` return an
`IntegrationResult` (`.val`, `.err`) and take a `limit` argument capping the
number of subintervals (default 50) - hitting the limit sets an
`MSL_EMAXITER`-family error code rather than silently truncating.

---

## Numerical differentiation (`msl.deriv`)

```mojo
from msl.deriv import deriv_central, deriv_forward, deriv_backward
from std.math import sin

def sin_fn(x: Float64) capturing -> Float64:
    return sin(x)

var r = deriv_central[sin_fn](1.0)   # d/dx sin(x) at x=1 ≈ cos(1)
print(r.val, r.err)
```

`deriv_central` (5-point, the usual default) needs `f` evaluable on both
sides of `x`; `deriv_forward`/`deriv_backward` (4-point) are for a boundary
where only one side is available. All three take an initial step `h`
(default `1e-4`) and return a `DerivResult` (`.val`, `.err`) - `deriv_central`
additionally re-estimates `h` once internally if rounding error dominates
truncation error, so the `h` you pass is a starting point, not a guarantee.

---

## Interpolation (`msl.interpolation`)

Three 1-D interpolators, all constructed from raw `Pointer[Float64,
origin]` `x`/`y` buffers plus a count - the interpolator does **not** copy
or take ownership of `xa`/`ya`, so they must outlive it:

```mojo
from std.memory.alloc import unsafe_alloc
from msl.interpolation import CubicSpline, AkimaSpline

var xa = unsafe_alloc[Float64](5)
var ya = unsafe_alloc[Float64](5)
xa[unsafe_offset=0]=0.0; xa[unsafe_offset=1]=1.0; xa[unsafe_offset=2]=2.0
xa[unsafe_offset=3]=3.0; xa[unsafe_offset=4]=4.0
ya[unsafe_offset=0]=0.0; ya[unsafe_offset=1]=1.0; ya[unsafe_offset=2]=4.0
ya[unsafe_offset=3]=9.0; ya[unsafe_offset=4]=16.0

var spline = CubicSpline(xa, ya, 5)
print(spline.eval(1.5).val)    # ≈ 2.25 (interpolates x^2)
print(spline.deriv(2.0).val)    # ≈ 4.0  (derivative of x^2)
xa.unsafe_free(); ya.unsafe_free()
```

- `LinearInterp` - piecewise linear, O(1) evaluation.
- `CubicSpline` - natural cubic spline, O(n) build, O(log n) evaluation
  (binary search over knots).
- `AkimaSpline` - Akima's local cubic method; less prone to overshoot near
  outliers than a global cubic spline, still O(log n) evaluation.

All three expose `.eval(x)`, `.deriv(x)`, `.deriv2(x)`, `.integral(a, b)`,
each returning an `InterpResult` (`.val`, `.errno`) - evaluating outside
`[xa[0], xa[n-1]]` sets `MSL_EDOM` rather than extrapolating silently.

---

## Root-finding and minimization (`msl.optimizer`)

```mojo
from msl.optimizer import root_brent, root_newton, min_brent
from std.math import sin, cos

def fn_(x: Float64) capturing -> Float64:
    return sin(x)

var r = root_brent[fn_](3.0, 4.0)     # sin(x) = 0 near pi
print(r.root, r.nit)                    # ≈ 3.14159...

def neg_cos(x: Float64) capturing -> Float64:
    return -cos(x)

var m = min_brent[neg_cos](-1.0, 0.5, 1.0)
print(m.x, m.fun)                        # ≈ 0.0, -1.0
```

Root-finders: `root_bisect` (bracketing, guaranteed convergence, slowest),
`root_brent` (bracketing + inverse quadratic interpolation, the usual
default), `root_falsepos` (regula falsi), `root_newton` (needs a
derivative, fastest when it converges), `root_secant` (derivative-free
Newton), `root_steffenson` (accelerated fixed-point). All bracketing
methods (`bisect`, `brent`, `falsepos`) require `f(a)` and `f(b)` to have
opposite signs and return a `RootResult` (`.root`, `.nit`, `.nfev`,
`.success`, `.errno`).

Minimizers: `min_brent` (parabolic interpolation with golden-section
fallback) and `min_golden` (pure golden section) both need a bracketing
triple `a < x < b` with `f(x) < f(a)` and `f(x) < f(b)`; both return a
`MinResult` (`.x`, `.fun`, `.nit`, `.success`, `.errno`).

For step-by-step control instead of a single call that iterates to
convergence internally, `RootFSolver`/`RootFDFSolver`/`MinFSolver` expose
an iterator-style struct you drive one `.iterate()` call at a time -
useful when you want to inspect or plot intermediate iterates. Convergence
test helpers (`root_test_interval`, `root_test_residual`,
`root_test_delta`, `min_test_interval`) and `min_find_bracket` (search
outward from a point for a valid bracketing triple) round out the module.

---

## ODE solving (`msl.ode`)

The right-hand side is a compile-time callback of a distinct shape from
[the usual one](#passing-a-function-to-msl) - it takes `t`, an input `y`
pointer, and an output `dydt` pointer to *write into* (no return value):

```mojo
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from msl.ode import ode_rk4, ode_rkf45
from std.math import cos

def rhs[o1: MutOrigin, o2: MutOrigin, //](
    t: Float64,
    y: Pointer[Float64, o1],
    dydt: Pointer[Float64, o2],
) capturing:
    # Harmonic oscillator: y0'' + y0 = 0, as a first-order system.
    dydt[unsafe_offset=0] = y[unsafe_offset=1]
    dydt[unsafe_offset=1] = -y[unsafe_offset=0]

var y = unsafe_alloc[Float64](2)
y[unsafe_offset=0] = 1.0; y[unsafe_offset=1] = 0.0   # y0(0)=1, y1(0)=0

_ = ode_rk4[rhs](0.0, 1.0, 0.01, y, 2)      # fixed step h=0.01
print(y[unsafe_offset=0], cos(1.0))          # y0(1) ≈ cos(1)

y[unsafe_offset=0] = 1.0; y[unsafe_offset=1] = 0.0
var r2 = ode_rkf45[rhs](0.0, 1.0, 0.1, y, 2, epsabs=1e-8, epsrel=1e-8)
print(r2.nsteps, "steps")
y.unsafe_free()
```

`y` is updated **in place** - the solution at `t1` is whatever `y` holds
when the call returns, there's no separate output buffer. `ode_rk4` is
fixed-step (you choose `h` and live with it); `ode_rkf45` is adaptive
(starts from `h0`, grows/shrinks step size to hit `epsabs`/`epsrel`, with
optional `hmin`/`hmax` clamps) and reports `.nsteps` taken in its
`OdeResult`.

---

## Random numbers and distributions (`msl.rng`, `msl.distributions`)

```mojo
from msl.rng import RNG, MT19937
from msl.distributions import gaussian, uniform, gamma, poisson, tdist, weibull

var rng = RNG[MT19937](MT19937(42), 42)

print(rng.uniform())              # uniform in [0,1)
print(gaussian(rng, 1.0))          # N(0, sigma=1)
print(gamma(rng, 2.0, 1.0))         # Gamma(shape=2, scale=1)
print(poisson(rng, 5.0))             # Poisson(lambda=5)
print(tdist(rng, 10.0))               # Student-t(nu=10)
print(weibull(rng, 1.0, 2.0))          # Weibull(scale=1, shape=2)
```

`RNG[T]` is parametric over any `RNGAlgorithm` implementation - `MT19937`
(Mersenne Twister) is the only one shipped, but you can plug in your own:

```mojo
struct MyRNG(RNGAlgorithm):
    def next(mut self) -> UInt64: ...
    def seed(mut self, s: UInt64): ...
```

Every distribution sampler takes `mut rng: RNG[T]` as its first argument
and is generic over `T: RNGAlgorithm`, so it works with any algorithm, not
just `MT19937`. Each of the 14 distributions (Gaussian, uniform,
exponential, gamma, beta, chi-squared, Poisson, Student-t, log-normal,
Weibull, binomial, negative binomial, Cauchy, Laplace) has both a sampler
and a `*_pdf` density function taking the same shape parameters.

---

## Vectors and matrices (`msl.vector`, `msl.matrix`, `msl.blas`)

`Vector` and `Matrix` are thin, BLAS-backed wrappers over a flat buffer -
GSL's `gsl_vector`/`gsl_matrix` shape, with an explicit `stride`
(`Vector`) or leading dimension `tda` (`Matrix`) so they can view a
strided slice of someone else's memory:

```mojo
from std.memory.alloc import unsafe_alloc
from msl.vector import Vector
from msl.matrix import Matrix
from msl.blas import blas_dot, blas_gemm

# Owning: allocates and frees its own buffer.
var x = Vector(3, initialize=True)
var y = Vector(3, initialize=True)
x[0]=1.0; x[1]=2.0; x[2]=3.0
y[0]=4.0; y[1]=5.0; y[2]=6.0
print(blas_dot(x, y))   # 32.0

# Non-owning: a zero-copy view over a buffer you manage (e.g. a NuMojo
# NDArray's backing pointer). The destructor does NOT free it.
var buf = unsafe_alloc[Float64](6)
_ = Vector(buf, 6)
buf.unsafe_free()

var A = Matrix(2, 3, initialize=True)
var B = Matrix(3, 2, initialize=True)
var C = Matrix(2, 2, initialize=True)
blas_gemm(A, B, C)   # C = A @ B
```

`msl.blas` wraps [mojoBLAS](https://github.com/shivasankarka/mojoBLAS)
Level 1 (`blas_dot`, `blas_nrm2`, `blas_asum`, `blas_axpy`, `blas_scal`,
`blas_copy`, `blas_swap`), Level 2 (`blas_gemv`), and Level 3 (`blas_gemm`)
- the same relationship GSL has to CBLAS. `Vector`/`Matrix` also carry
their own scalar-loop implementations of the common ops (`vector_add`,
`matrix_transpose`, ...) for cases too small to bother dispatching to BLAS.

---

## Dense linear algebra (`msl.linalg`)

Like `msl.integration`/`msl.deriv`, this layer is **not** built on
`Vector`/`Matrix` - it operates directly on `Pointer[Float64, origin]` +
`lda` (leading dimension), matching GSL's C-level LAPACK-style API. Build a
`Matrix` first if you want the higher-level type, and pass
`.mut_ptr()`/`.tda` through:

```mojo
from msl.linalg import lu_decomp, lu_solve, cholesky_decomp, cholesky_solve, qr_decomp, eigen_jacobi
```

- `lu_decomp`/`lu_svx`/`lu_solve` - LU with partial pivoting (`piv` array
  holds raw pivot indices, not a full permutation - see
  [Permutations](#permutations-mslpermutation) if you need one);
  `lu_det`/`lu_lndet` and `lu_invert` build on the factorization.
- `cholesky_decomp`/`cholesky_svx`/`cholesky_solve` - for symmetric
  positive-definite `A`; cheaper than LU when it applies, fails (non-zero
  return / `MSL_EDOM`) if `A` isn't actually SPD.
- `qr_decomp`/`qr_svx`/`qr_solve` - Householder QR; more numerically
  robust than LU-based least squares for ill-conditioned systems.
- `eigen_jacobi` - cyclic Jacobi eigensolver for symmetric matrices; O(n³)
  per sweep, fine for small-to-medium `n`, not a substitute for a
  Householder-tridiagonalization-based solver on large matrices.

---

## Polynomials (`msl.poly`)

Also `Pointer`-based, not `Vector`-based:

```mojo
from msl.poly import poly_eval, poly_solve_quadratic, poly_solve_cubic

# c[0] + c[1]*x + c[2]*x^2 via Horner's method
var y = poly_eval(coeffs_ptr, 3, 2.0)

var roots = poly_solve_quadratic(1.0, -3.0, 2.0)   # x^2 - 3x + 2 = 0 -> {1, 2}
print(roots.nroots, roots.x0, roots.x1)
```

`poly_solve_quadratic`/`poly_solve_cubic` return the **real** roots only
(`QuadraticRoots`/`CubicRoots`, with `.nroots` telling you how many of
`.x0`/`.x1`/(`.x2`) are populated) - complex-conjugate pairs are not
reported as complex numbers, matching GSL's real-roots-only convenience
functions (`gsl_poly_solve_quadratic`/`_cubic`), not the general complex
solver. `poly_dd_init`/`poly_dd_eval`/`poly_dd_taylor` implement Newton
divided-difference interpolation, giving you a polynomial through a set of
`(x, y)` points and, via `poly_dd_taylor`, its Taylor expansion about a
point.

---

## Permutations (`msl.permutation`)

```mojo
from msl.permutation import Permutation, permutation_init, permutation_next
```

A `Permutation` is a fixed-size array of `Int`-like indices. `Permutation(n)`
allocates and `permutation_init` sets it to the identity `[0, 1, ..., n-1]`;
`permutation_next` steps it to the lexicographically next permutation
in-place (returns false once past the last one), matching GSL's
`gsl_permutation_next` for exhaustive enumeration.

---

## Statistics (`msl.statistics`)

Also `Pointer` + stride based, matching GSL's `gsl_stats` functions
exactly (including the `stride` argument, so you can compute statistics
over a column of a row-major matrix without copying it out):

```mojo
from msl.statistics import stats_mean, stats_variance, stats_sd, stats_spearman
```

- Central moments: `stats_mean`, `stats_variance`/`stats_sd` (and
  `_with_fixed_mean` variants when you already know the population mean),
  `stats_skew`, `stats_kurtosis`.
- Association: `stats_covariance`, `stats_correlation`,
  `stats_lag1_autocorrelation`.
- Order statistics - **all require pre-sorted data**, the function name
  says so (`stats_median_from_sorted_data`,
  `stats_quantile_from_sorted_data`, `stats_trmean_from_sorted_data`,
  `stats_gastwirth_from_sorted_data`, `stats_Sn_from_sorted_data`,
  `stats_Qn_from_sorted_data`) - sort your buffer first, MSL does not sort
  for you.
- Robust scale (Rousseeuw & Croux 1993): `stats_mad`/`stats_mad0` (median
  absolute deviation), `stats_Sn_from_sorted_data`, `stats_Qn_from_sorted_data`
  - more resistant to outliers than `stats_sd` when that matters.
- Weighted counterparts (`stats_wmean`, `stats_wvariance`, `stats_wskew`,
  `stats_wkurtosis`, ...) mirror the unweighted functions one-for-one, with
  a `w` weight buffer + `wstride` in front of the data buffer + `stride`.
- `stats_ttest`/`stats_pvariance` - two-sample t-test and pooled variance.
- `stats_spearman` - Spearman rank correlation.

---

## Errors

MSL follows GSL's C-style error model, not exceptions: every function
either returns an `Int` error code directly, or carries one in an `.errno`
field on its result struct. `MSL_SUCCESS` (0) means no error; look up a
non-zero code against the `MSL_E*` constants in `msl.core.errno` (mirroring
GSL's `GSL_E*`: `MSL_EDOM` domain error, `MSL_EINVAL` invalid argument,
`MSL_EMAXITER` iteration limit hit, `MSL_EZERODIV` division by zero,
`MSL_ESING` singular matrix, and so on). There is no MSL-specific exception
type to catch - a `def ... raises` signature in this codebase almost always
means "calls something that itself uses `raises`" (e.g. Mojo's `String`
formatting), not "reports domain errors this way."

---

## Appendix: relationship to GSL

MSL is a close port of selected GSL routines into Mojo - same algorithms,
same default tolerances and iteration limits, same flat-buffer/leading-
dimension calling convention, same `errno`-based error reporting. It is
**not** a 1:1 API-compatible binding: names are `snake_case`d
(`gsl_sf_bessel_J0` → `bessel_j0`/`bessel_J0`), result structs replace
GSL's output-parameter-plus-return-code idiom where that's more natural in
Mojo, and only a subset of GSL's surface is ported (see the [README's
Roadmap](../README.md#roadmap) and the section below for what isn't here).
MSL is licensed GPL-3.0-or-later, matching GSL, since large parts of it are
a derivative work - see the [Developer Guide](developer_guide.md#file-header)
for what that means for contributing new files.

## Appendix: what is not here yet

- FFT (planned: pure-Mojo Cooley-Tukey, ported from SciJo's implementation).
- Complete Bessel spherical functions (`j_n`, `y_n`).
- Elliptic integrals (K, E, Π) and hypergeometric functions (2F1, 1F1).
- Implicit ODE solvers (RK4-implicit, BDF) - only explicit RK4/RKF45 today.
- Sparse matrices - `msl.matrix`/`msl.linalg` are dense-only.

See the [README](../README.md#roadmap) for the up-to-date list.
