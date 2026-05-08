# SPDX-License-Identifier: GPL-3.0-or-later

# ===----------------------------------------------------------------------=== #
# MSL (Mojo Scientific Library)
#
# Derived from GNU Scientific Library (GSL)
# Original file: gsl_rng.h, mt19937.c
#
# Original authors:
# Copyright (C) 1996–2007 James Theiler, Brian Gough
# Modified by Isaku Yamada
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
"""
Random number generator: Mersenne Twister (MT19937).

This is a pure Mojo implementation of the Mersenne Twister algorithm.
Based on the GSL implementation which is based on the original
algorithm by Makoto Matsumoto and Takuji Nishimura.
"""

from std.memory import UnsafePointer


# ===----------------------------------------------------------------------=== #
# MT19937 constants
# ===----------------------------------------------------------------------=== #

comptime MT_N: Int = 624
comptime MT_M: Int = 397
comptime MT_MATRIX_A: UInt64 = 0x9908B0DF
comptime MT_UPPER_MASK: UInt64 = 0x80000000
comptime MT_LOWER_MASK: UInt64 = 0x7FFFFFFF


# ===----------------------------------------------------------------------=== #
# MT19937 state
# ===----------------------------------------------------------------------=== #


struct MTState(Copyable, Movable):
    """Mersenne Twister state."""

    var state: UnsafePointer[UInt64, MutExt]
    var idx: Int

    def __init__(out self):
        self.state = alloc[UInt64](MT_N)
        self.idx = MT_N

    def __del__(deinit self):
        self.state.free()

    def __getitem__(self, i: Int) -> UInt64:
        return self.state[i]

    def __setitem__(mut self, i: Int, value: UInt64):
        self.state.store(i, value)

    def __len__(self) -> Int:
        return MT_N


# ===----------------------------------------------------------------------=== #
# MT19937 functions
# ===----------------------------------------------------------------------=== #


def mt19937_init(mut state: MTState, seed: UInt64):
    """Initialize MT19937 with seed."""
    state[0] = seed
    for i in range(1, MT_N):
        var prev = state.state[i - 1]
        state.state.store(
            i,
            (1812433253 * (prev ^ (prev >> 30)) + UInt64(i))
            % UInt64(0x100000000),
        )
    state.idx = MT_N


def mt19937_get(mut state: MTState) -> UInt64:
    """Generate next MT19937 random value."""
    if state.idx >= MT_N:
        var i: Int = 0
        while i < MT_N - MT_M:
            var y = (state.state[i] & MT_UPPER_MASK) | (
                state.state[i + 1] & MT_LOWER_MASK
            )
            var idx = i + MT_M
            state.state.store(
                i, state.state[idx] ^ (y >> 1) ^ MT_MATRIX_A * (y & 1)
            )
            i += 1

        while i < MT_N - 1:
            var y = (state.state[i] & MT_UPPER_MASK) | (
                state.state[i + 1] & MT_LOWER_MASK
            )
            var idx = i + MT_M - MT_N
            state.state.store(
                i, state.state[idx] ^ (y >> 1) ^ MT_MATRIX_A * (y & 1)
            )
            i += 1

        y = (state.state[MT_N - 1] & MT_UPPER_MASK) | (
            state.state[0] & MT_LOWER_MASK
        )
        var idx = MT_M - 1
        state.state.store(
            MT_N - 1, state.state[idx] ^ (y >> 1) ^ MT_MATRIX_A * (y & 1)
        )

        state.idx = 0

    var y2 = state.state[state.idx]
    state.idx += 1

    y2 ^= y2 >> 11
    y2 ^= (y2 << 7) & 0x9D2C5680
    y2 ^= (y2 << 15) & 0xEFC60000
    y2 ^= y2 >> 18

    return y2


def mt19937_get_double(mut state: MTState) -> Float64:
    """Generate uniform double in [0, 1)."""
    return Float64(mt19937_get(state)) / Float64(0x100000000)


# ===----------------------------------------------------------------------=== #
# RNG wrapper
# ===----------------------------------------------------------------------=== #


struct RNG(Copyable, Movable):
    """Random number generator using MT19937."""

    var state: MTState
    var seed_val: UInt64
    var name: String

    def __init__(out self, seed: UInt64 = 5489):
        self.state = MTState()
        self.seed_val = seed
        self.name = "mt19937"
        mt19937_init(self.state, seed)

    def __del__(deinit self):
        _ = self.state

    def set_seed(mut self, seed: UInt64):
        """Set the seed."""
        self.seed_val = seed
        mt19937_init(self.state, seed)

    def get(mut self) -> UInt64:
        """Get next unsigned 64-bit integer."""
        return mt19937_get(self.state)

    def uniform(mut self) -> Float64:
        """Get uniform random double in [0, 1)."""
        return mt19937_get_double(self.state)

    def uniform_pos(mut self) -> Float64:
        """Get uniform random double in (0, 1)."""
        var x = self.uniform()
        while x == 0.0:
            x = self.uniform()
        return x

    def uniform_int(mut self, n: Int) -> Int:
        """Get uniform random integer in [0, n)."""
        return Int(self.uniform() * Float64(n))
