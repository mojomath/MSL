# MSL - Mojo Scientific Library

A pure Mojo implementation of scientific computing functions, derived from the GNU Scientific Library (GSL).

## Motivation

Porting GSL is a huge task, but it's much better than writing crazy bindings lol. I also get exposed to more functions that might be helpful in my research, and it helps me learn the math and how to do numerical scientific computing in general (people have come up with crazy tricks!).

I ported a lot of GSL integration routines, special functions and solvers like `hermv`, ODE solvers like `rkf45`, etc. for my own research project in Mojo. I also ported many statistics routines for Scijo and Statmojo. So I thought it's better to bundle it all together like the GSL library.

I don't plan to port everything from GSL - perhaps if I get time in the future, I'll do more. I'm porting just enough for our libraries: Scijo, Statmojo, and HEPJo. Check out our org [MojoMath](https://github.com/mojomath) for more details.

## Overview

MSL provides the building blocks for scientific computing in Mojo:

- **Vectors & Matrices** - Core objects for linear algebra and more. 
- **Random Numbers** - RNGs and probability distributions (Gaussian, uniform, exponential, etc.)
- **Special Functions** - Airy, Bessel, and more
- **Permutations** - Combinatorial algorithms
- **Constants & Utilities** - Mathematical constants, error handling

## Installation

### With pixi (Recommended)

Add to your `pixi.toml`:

```toml
[workspace]
preview = ["pixi-build"]

[package]
name = "your_project"
version = "0.1.0"

[package.build]
backend = {name = "pixi-build-mojo", version = "0.*"}

[dependencies]
mojo = ">=25.7.0,<26"
msl = { git = "https://github.com/shivasankarka/MojoScientificLibrary.git", branch = "main"}
```

Then run `pixi install`.

### Direct clone

```bash
git clone https://github.com/shivasankarka/MojoScientificLibrary.git
cd MojoScientificLibrary
pixi run mojo package msl
```

Move the generated `msl.mojopkg` to your project or `$CONDA_PREFIX/lib/` in pixi envs. 

I'll add this to the Mojo community repo later to make installation easier.

## Usage

```mojo
from msl.sf import airy_ai, bessel_j0
from msl.rng import RNG
from msl.distributions import gaussian

# Special functions
var ai = airy_ai(0.0).val        # 0.355028053887835
var j0 = bessel_j0(1.0).val      # 0.765197686557967

# Random numbers
var rng = RNG()
var x = gaussian(rng, 0.0, 1.0)  # random from N(0,1)
```

All sf functions return `SFSResult` with `.val` (computed value) and `.err` (error estimate).

## API

```mojo
from msl.sf import airy_ai, airy_bi, bessel_j0, bessel_j1, ...
from msl.vector import Vector
from msl.matrix import Matrix
from msl.rng import RNG
from msl.distributions import uniform, gaussian, exponential
from msl.permutation import Permutation
```

## Status

| Module | Status |
|--------|--------|
| sf/airy | ✅ |
| sf/bessel | ✅ |
| vector | ✅ |
| matrix | ✅ |
| rng | ✅ |
| distributions | ✅ |
| permutation | ✅ |

## Roadmap

- [ ] Gamma and error functions
- [ ] Legendre functions
- [ ] Elliptic integrals
- [ ] Hypergeometric functions
- [ ] Numerical integration
- [ ] ODE solvers
- [ ] Optimize with Mojo-specific tricks! The fun part!
 
## Contributors

All contributions are welcome. Let's make science easy together for everyone :)

I probably won't be able to port everything, so if anyone's interested, please have a go at it and make PRs - I'll be happy to discuss and merge.

## License

GPL-3.0-or-later. See [LICENSE](LICENSE) for details.

This project is derived from the GNU Scientific Library (GSL). The original GSL code is Copyright (C) 1996-2007 Gerard Jungman, Brian Gough and contributors.
