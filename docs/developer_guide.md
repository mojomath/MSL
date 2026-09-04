# MSL Developer Guide

This guide covers the conventions actually in use across the MSL codebase.
It documents what is here, not what another Mojo scientific library does -
see the note at the top of each section if a rule differs from what you
might expect from NuMojo, SciJo, or mojoBLAS.

---

## Project Structure

```
msl/
├── __init__.mojo         # Top-level package re-exports (the full public API)
├── core/                 # Constants, error codes (MSL_*), pointer/math utilities
├── sf/                   # Special functions (Airy, Bessel, Gamma, Beta, Erf, Legendre, ...)
├── integration/           # Gauss-Kronrod quadrature (QNG, QK15-QK61, QAG, QAGS)
├── deriv/                # Numerical differentiation
├── interpolation/         # Linear, cubic spline, Akima interpolation
├── optimizer/             # Root-finding and minimization
├── ode/                   # RK4 and RKF45 ODE solvers
├── distributions/         # Probability distributions
├── rng/                   # MT19937 and the RNGAlgorithm trait
├── vector/, matrix/        # BLAS-backed dense vector/matrix types
├── blas/                  # Thin wrapper over mojoBLAS
├── linalg/                # LU, Cholesky, QR, symmetric eigensolver
├── permutation/            # Permutation arrays
├── poly/                  # Polynomial evaluation, interpolation, root solving
└── statistics/             # Descriptive statistics
tests/
├── tests.sh               # Runs every tests/test_*.mojo file
└── test_*.mojo             # One (or more, e.g. sf/sf_m11) file per module
```

Each submodule follows the same internal layout:

- `__init__.mojo` - public API re-exports, no implementation of its own.
- One or more implementation files.

MSL is **scalar-first**: every function operates on `Float64` (occasionally
`Int`), never on arrays or NDArrays. Array-level operations belong in SciJo,
which uses MSL as its low-level numeric backend - the same relationship GSL
has with SciPy.

---

## File Header

Nearly all of MSL is either a direct port of a [GNU Scientific
Library](https://www.gnu.org/software/gsl/) (GSL) routine, or original code
written to sit alongside those ports. GSL is GPL-3.0-or-later, and a ported
file is a derivative work under copyright law - it cannot be relicensed
permissively. **The whole package is GPL-3.0-or-later** (see
[LICENSE](../LICENSE)); every `.mojo` file starts with the same banner:

```
# SPDX-License-Identifier: GPL-3.0-or-later

# ===----------------------------------------------------------------------=== #
# MSL (Mojo Scientific Library)
#
# Derived from GNU Scientific Library (GSL)
# Original file: <path in GSL, e.g. specfunc/gamma.c>
#
# Original authors:
# Copyright (C) <years> <GSL authors>
#
# Modifications:
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
```

Use the full 12-line GPL notice (warranty disclaimer + license-text
pointer), not the shorter 4-line "free software" clause alone - a handful of
files used to have the short form and were standardized to the full one.

For a file with no GSL-derived content (a package `__init__.mojo`, a small
original helper like `core/types.mojo`), drop the "Derived from .../Original
file/Original authors" lines and just keep the project name and your own
copyright line - see `msl/optimizer/utility.mojo` for the shape.

Immediately after the closing separator, **with no blank line**, comes the
module docstring:

```
"""
One-line free-text description of what this file provides.

Optional further paragraphs - context on the algorithm, what's ported from
where, usage notes.
"""
```

> **Different from NuMojo/mojoBLAS**: MSL's module docstring does not use a
> `Name (dotted.path).` title line with a matching `===` underline. It is
> just a plain sentence (or a couple of paragraphs), the same way a function
> docstring's summary line reads.

---

## Docstring Format

Sections appear in this order when present: `Parameters:` (compile-time
parameters), `Args:` (runtime arguments), `Returns:`, `Raises:`, `Notes:`,
`References:`, `Examples:`. Use `Args:`, never `Arguments:`.

MSL is **not** strict about every function carrying every applicable
section - plenty of short, self-explanatory accessors get a one-line
docstring and nothing else:

```mojo
def size1(self) -> Int:
    """Return number of rows."""
    return self.s1
```

For anything less obvious - especially a public top-level function with
several arguments - document `Args:`/`Returns:`:

```mojo
def blas_dot(x: Vector, y: Vector) -> Float64:
    """Dot product x · y.

    Args:
        x: First vector.
        y: Second vector (must have same size as x).

    Returns:
        Scalar dot product, or 0.0 if sizes differ.
    """
```

### Struct docstrings

```mojo
struct Matrix(Copyable, Movable):
    """2D dense matrix."""

    var s1: Int
    """Number of rows."""
    var s2: Int
    """Number of columns."""
    var tda: Int
    var data: Pointer[Float64, MutExt]
    var owner: Bool
```

Give a field a docstring when its meaning isn't obvious from its name and
type (`s1`/`s2` above); an internal bookkeeping field like `tda` or `owner`
is fine without one. This is looser than "every field must have one" - match
what you see near the struct you're editing.

---

## Error Handling

> **Different from SciJo/NuMojo**: MSL does not raise a custom error struct.
> Following GSL's own C API, functions report failure through **return
> values**, not exceptions:
>
> - Special functions return an `SFSResult` with `.val`, `.err`, and
>   `.errno` (0 = `MSL_SUCCESS`); check `.errno` against the `MSL_E*`
>   constants in `msl.core.errno`, not `.val`.
> - Most other functions return an `Int` error code directly, or write it
>   into a result struct's `errno` field.
> - `msl_error`/`msl_assert` (in `msl.core.errno`) exist for the rare case
>   that genuinely needs to raise, but real `raise Error(...)` usage is
>   almost nonexistent in this codebase - don't reach for it as the default.

Because of this, function docstrings essentially never carry a `Raises:`
section; document the error path through `.errno`/return-code semantics in
`Returns:` or a short note instead.

---

## Imports

> **Different from GSL-derived-vs-original file distinctions elsewhere**:
> import style does not depend on a file's license header - it's uniform.

All internal imports are **absolute**, never relative:

```mojo
from msl.integration.qk21 import qk21   # correct
from .qk21 import qk21                  # wrong - don't do this
```

Imports are grouped and banner-labeled, stdlib first:

```mojo
# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.math import abs, sqrt
from std.memory import Pointer

# ===----------------------------------------------------------------------=== #
# MSL
# ===----------------------------------------------------------------------=== #
from msl.core.const import MSL_DBL_EPSILON
from msl.integration.qk21 import qk21
```

`mojoblas.*` imports (used only by `msl.blas`) count as external, grouped
with any third-party dependency, not under the `MSL` banner.

Don't hand-sort these - run the organizer script (see below), which also
dedupes `from X import (...)` blocks and drops unused imports (except in
`__init__.mojo` re-export files, which are left alone since "unused" there
usually means "re-exported").

---

## Comments

- Don't write comments the code and docstrings already make obvious.
- Use `# TODO: Capitalized sentence.` for planned work - the checker script
  flags a `TODO`/`FIXME` that doesn't read as a real sentence.
- For inline section separators within a file (e.g. splitting out a
  Chebyshev coefficient table, or grouping a struct's methods), use the same
  3-line banner as the file header, indented to match the surrounding code:

```
# ===----------------------------------------------------------------------=== #
# Section Title
# ===----------------------------------------------------------------------=== #
```

---

## Testing

Tests live in `tests/`, one file per module (`test_sf.mojo`,
`test_statistics.mojo`, ...) using `std.testing.TestSuite`, which
auto-discovers every `test_*` function in the file:

```mojo
from std.testing import TestSuite

def test_something() raises:
    var result = some_function(1.0)
    assert tolerance(result.val, 0.5, 1e-10)
    print("test_something: PASSED")

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
```

Guidelines:

- Reference values come from a trusted external source (GSL itself, or
  `scipy.special`/`scipy.stats` when porting from GSL isn't practical) -
  never hand-computed approximations.
- When a function has more than one code path selected by the input range
  (e.g. a Chebyshev-series branch for small `x` and an asymptotic branch for
  large `x`), test **each branch separately**. A bug can live entirely in a
  branch that a single "obvious" test value never reaches - see the
  `bessel_j1`/`bessel_y1` large-`x` regression tests in `test_sf.mojo` for
  an example: `bessel_j1(1.0)` alone never touched the corrupted coefficient
  table that only the `y > 4.0` branch used.
- Run the full suite with:

```sh
pixi run tests
```

---

## Common Commands

| Task | Command |
|---|---|
| Format all code | `pixi run format` |
| Package the library | `pixi run package` |
| Run all tests | `pixi run tests` |
| Check files against these standards | `python3 scripts/check_mojo_standards.py msl` |
| Sort/dedupe/absolutize imports | `python3 scripts/organize_mojo_imports.py msl` |

`check_mojo_standards.py` is reporting-only - it never edits a file.
`organize_mojo_imports.py` rewrites files in place by default; pass
`--check` for a dry run that exits non-zero if changes are needed, or
`--diff` to preview them.

---

## Checklist Before Submitting

- [ ] Every new file has the GPL-3.0-or-later header (banner form for a GSL
      port with attribution; the shorter original-file form otherwise).
- [ ] Every new file has a module-level docstring, no blank line before it.
- [ ] New internal imports are absolute (`msl.xxx.yyy`), grouped under
      `Stdlib`/`MSL` banners.
- [ ] Public functions with non-obvious arguments document `Args:`/
      `Returns:`.
- [ ] `__init__.mojo` re-exports every new public symbol from the submodule.
- [ ] Tests exist for new functionality, and - if the function has more than
      one code path - cover each path, not just the default/small-input one.
- [ ] `pixi run format` has been run.
- [ ] `pixi run package` has been run and compiles with zero errors.
- [ ] `pixi run tests` has been run and every test passes.
- [ ] `python3 scripts/check_mojo_standards.py msl` reports no findings (or
      only pre-existing ones you're not touching).
- [ ] `python3 scripts/organize_mojo_imports.py msl --check` reports no
      changes needed.
