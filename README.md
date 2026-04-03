# MSL - Mojo Scientific Library

A pure Mojo implementation of scientific computing functions, derived from the GNU Scientific Library (GSL).

## Motivation

Porting GSL is a huge task, but it's much better than writing crazy bindings lol. I also get exposed to more functions that might be helpful in my research, and it helps me learn the math and how to do numerical scientific computing in general (People have come up with crazy tricks!). 

I ported a lot of GSL integration routines, special functions and solvers like `hermv`, ode solvers like `rkf45` etc for my own research project written in Mojo. I also ported many statistics routines for Scijo and Statmojo. So I thought it's better to bundle it all together like the GSL library. 

I don't plan to port everything from GSL - perhaps if I get time in the future, I'll do more. I'm porting just enough for our libraries: Scijo, Statmojo, and HEPJo. Check out our org [MojoMath](https://github.com/mojomath) for more details on our libraries :)

## Overview

MSL aims to port the basic functions needed in day-to-day scientific computing from GSL.

## Features

### Special Functions (sf)
- **Airy functions**: Ai(x), Bi(x), scaled versions, and derivatives
- **Bessel functions**: J₀, J₁, Y₀, Y₁, I₀, I₁, K₀, K₁, and spherical variants

### Core Modules
- Vectors and matrices with memory management
- Random number generators (RNG)
- Probability distributions
- Permutations
- Error handling and constants

## Installation

### Method 1: Git with pixi (Recommended)

Add to your `pixi.toml`:

```toml
[workspace]
preview = ["pixi-build"]

[package]
name = "your_project_name"
version = "0.1.0"

[package.build]
backend = {name = "pixi-build-mojo", version = "0.*"}

[dependencies]
mojo = ">=25.7.0,<26"
msl = { git = "https://github.com/shivasankarka/MojoScientificLibrary.git", branch = "main"}
```

Then run:
```bash
pixi install
```

### Method 2: Clone directly
```bash
git clone https://github.com/shivasankarka/MojoScientificLibrary.git
cd msl
bash tests/tests.sh
```

## Usage

```mojo
from msl.sf import airy_ai, bessel_j0, SFSResult

# Airy function Ai
var result = airy_ai(0.0)
print(result.val)  # 0.355028053887835

# Bessel function J0
var j0 = bessel_j0(1.0)
print(j0.val)      # 0.765197686557967
```

All functions return `SFSResult` containing:
- `val`: computed value
- `err`: error estimate

## API

```mojo
from msl.sf import (
    airy_ai,
    airy_bi,
    airy_ai_scaled,
    airy_bi_scaled,
    bessel_j0,
    bessel_j1,
    bessel_y0,
    bessel_y1,
    bessel_i0_scaled,
    bessel_i1_scaled,
    bessel_k0_scaled,
    bessel_k1_scaled,
)

from msl.vector import Vector
from msl.matrix import Matrix
from msl.rng import RNG
from msl.distributions import uniform, gaussian, exponential
from msl.permutation import Permutation
```

## Status

**Version**: 0.1.0 | **Tests**: 78 passing

| Module | Status |
|--------|--------|
| sf/airy | ✅ Complete |
| sf/bessel | ✅ Complete |
| vector | ✅ Complete |
| matrix | ✅ Complete |
| rng | ✅ Complete |
| distributions | ✅ Complete |
| permutation | ✅ Complete |

## Roadmap

- [ ] Gamma and error functions
- [ ] Legendre functions
- [ ] Elliptic integrals
- [ ] Hypergeometric functions
- [ ] Chebyshev polynomials
- [ ] Numerical integration
- [ ] ODE solvers
- [ ] Go through all existing routines and optimize them with mojo specific optimizations. 

## Contributors

All contributions are always welcome. Let's make science easy together for everyone :) 

I probably won't be able to port everything, so if anyone's interested, please have a go at it and make PRs - I'll be happy to discuss and merge.

## License

GPL-3.0-or-later. See [LICENSE](LICENSE) for details.

This project is derived from the GNU Scientific Library (GSL). The original GSL code is Copyright (C) 1996-2007 Gerard Jungman, Brian Gough and contributors.
