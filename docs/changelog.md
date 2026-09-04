# Changelog

All notable changes to this project will be documented in this file.

The format is based on "Keep a Changelog" and follows Semantic Versioning.

## [v0.2.0] - 2026-09-04

### Added
- **Linear algebra** (`msl.linalg`, new module):
  - `lu_decomp`, `lu_svx`, `lu_solve`, `lu_det`, `lu_lndet`, `lu_invert` - LU decomposition with partial pivoting and downstream solves.
  - `cholesky_decomp`, `cholesky_svx`, `cholesky_solve` - Cholesky decomposition for symmetric positive-definite matrices.
  - `qr_decomp`, `qr_svx`, `qr_solve` - Householder QR decomposition and solve.
  - `eigen_jacobi` - Symmetric eigenvalue/eigenvector solver (cyclic Jacobi).
- **Polynomials** (`msl.poly`, new module):
  - `poly_eval`, `poly_eval_derivs` - Horner-form evaluation and derivatives.
  - `poly_dd_init`, `poly_dd_eval`, `poly_dd_taylor` - Newton divided-difference interpolation.
  - `poly_solve_quadratic`, `poly_solve_cubic` - Real/complex root solving.
- **Statistics** (`msl.statistics`, new module):
  - `stats_mean`, `stats_variance`, `stats_sd`, `stats_skew`, `stats_kurtosis` and fixed-mean variants.
  - `stats_covariance`, `stats_correlation`, `stats_lag1_autocorrelation`.
  - Order statistics on pre-sorted data: `stats_median_from_sorted_data`, `stats_quantile_from_sorted_data`, `stats_trmean_from_sorted_data`, `stats_gastwirth_from_sorted_data`, `stats_min`/`stats_max`/`stats_minmax` and index variants, `stats_select`.
  - Robust scale estimators (Rousseeuw & Croux): `stats_mad`/`stats_mad0`, `stats_Sn_from_sorted_data`/`stats_Sn0_from_sorted_data`, `stats_Qn_from_sorted_data`/`stats_Qn0_from_sorted_data`.
  - Weighted counterparts of the moment functions: `stats_wmean`, `stats_wvariance`, `stats_wsd`, `stats_wskew`, `stats_wkurtosis`, etc.
  - `stats_ttest`, `stats_pvariance` - two-sample t-test and pooled variance.
  - `stats_spearman` - Spearman rank correlation.
- **Optimizer** (`msl.optimizer`):
  - `root_falsepos` - False position (regula falsi) root-finding.
  - `root_steffenson` - Steffenson's method (accelerated fixed-point iteration).
  - `RootFSolver`, `RootFDFSolver`, `MinFSolver` - iterator-style solver structs for step-by-step convergence control.
  - `root_test_interval`, `root_test_residual`, `root_test_delta`, `min_test_interval`, `min_find_bracket` - convergence test and bracket-search helpers, plus the `MinBracketResult` struct.
- **Special functions** (`msl.sf`):
  - `psi`, `psi_n` - digamma and polygamma functions.
  - `gamma_inc`, `gamma_inc_P`, `gamma_inc_Q` - incomplete gamma functions.
  - `beta_inc` - regularized incomplete beta function.
  - Filled out the Bessel family: `bessel_j0`/`bessel_j1`/`bessel_y0`/`bessel_y1`/`bessel_Jn`/`bessel_Yn`/`bessel_In`, plus the remaining Airy, Gamma, and Erf variants (`airy_bi`, `airy_*_deriv*`, `double_factorial`, `erf_Z`, `erf_Q`, `hazard`, `log_erfc`, ...).
- **Testing**: regression tests for the `bessel_j1`/`bessel_y1` large-`x` (`y > 4.0`) asymptotic amplitude/phase branch (`test_bessel_j1_large_x`, `test_bessel_y1_large_x`, and further large/negative-`x` cases), added alongside the coefficient-table fix below.
- **Documentation**: `docs/MANUAL.md` (a prose tour of every module, framed for a GSL user), `docs/developer_guide.md` (MSL's actual header/docstring/import/testing conventions), and this changelog.
- `scripts/check_mojo_standards.py` and `scripts/organize_mojo_imports.py`, adapted from the sibling mojoBLAS/NuMojo tooling to MSL's own conventions (see Developer Guide).

### Changed
- **Relicensed the whole package to GPL-3.0-or-later.** Files that were originally Apache-2.0 (`__init__.mojo` re-export shims, `core/types.mojo`, `blas/blas.mojo`, `optimizer/utility.mojo`) are now GPL-3.0-or-later like the GSL-derived files, since the project is distributed as one combined GPL work; every file now carries the full `# ===` banner with the complete 12-line GPLv3 notice (warranty disclaimer + license-text pointer), replacing a mix of full and truncated 4-line notices.
- Standardized import ordering across the whole package: internal imports are always absolute (`msl.integration.qk21`, never `.qk21`), grouped under `# ===` / `Stdlib` / `MSL` banners.
- Updated every module and the README to Mojo 1.0.0 (from `1.0.0b1`/`1.0.0b2`): `Array` in place of `InlineArray` in several `msl.sf` tables, `materialize[...]()` where a `comptime` array is passed to a runtime call, `unsafe_offset=` in place of positional `__getitem__`, `unsafe_alloc`/`unsafe_free` in place of the deprecated positional `alloc`/`.free()`, and `Pointer`/`MutUntrackedOrigin` in place of the removed `UnsafePointer`/`MutExternalOrigin`.
- Fixed pointer access patterns across `msl.core.block`, `msl.vector`, `msl.matrix`, `msl.linalg`, `msl.interpolation`, `msl.permutation`, and `msl.statistics` to match the above.

### Fixed
- **`bessel_j1`/`bessel_y1` large-`x` amplitude/phase (`msl/sf/bessel.mojo`)**: the `bth1_c` Chebyshev coefficient table used for `y > 4.0` was corrupted - declared as `Array[Float64, 24]` but initialized with only 23 elements (a compile error under Mojo 1.0's stricter array-literal checking), and even at the right length the values did not match GSL's `bth1_data` table at all. Replaced with the correct 24-value table from GSL's `bessel_amp_phase.c`, verified against a local GSL 2.8 checkout. This affected `bessel_J1`, `bessel_Y1`, and their aliases `bessel_j1`/`bessel_y1` for all `x` with `|x| > 4`; values for `|x| <= 4` were unaffected (different code path).

### Removed
- No removals in this release.

### Security
- No security-related changes in this release.

## [v0.1.0] - initial release

- Initial public release: `msl.sf` (Airy, core Bessel/Gamma/Beta/Erf/Legendre), `msl.integration` (QNG, QK15-QK61, QAG, QAGS), `msl.deriv`, `msl.interpolation`, `msl.optimizer` (bisection, Brent, Newton, secant, Brent/golden minimization), `msl.ode` (RK4, RKF45), `msl.distributions`, `msl.rng` (MT19937), `msl.vector`/`msl.matrix` (BLAS-backed), `msl.blas`, `msl.permutation`, `msl.core`.
