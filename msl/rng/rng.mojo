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
Random number generator infrastructure and MT19937 implementation.

Defines the `RNGAlgorithm` trait and the parametric `RNG[T]` wrapper.
`MT19937` is the default algorithm.
"""

from std.memory import UnsafePointer


# ===----------------------------------------------------------------------=== #
# RNGAlgorithm trait
# ===----------------------------------------------------------------------=== #


trait RNGAlgorithm(Copyable, Movable, ImplicitlyDestructible):
    """Interface for RNG algorithms.

    Implementors provide raw 64-bit integer generation and seeding.
    The `RNG[T]` wrapper adds the uniform/integer convenience API on top.
    """

    def next(mut self) -> UInt64:
        ...

    def seed(mut self, s: UInt64):
        ...


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
    """Mersenne Twister internal state buffer."""

    var state: UnsafePointer[UInt64, MutExt]
    var idx: Int

    def __init__(out self):
        self.state = alloc[UInt64](MT_N)
        self.idx = MT_N

    def __init__(out self, *, deinit take: Self):
        self.state = take.state
        self.idx = take.idx

    def __init__(out self, *, copy: Self):
        self.state = alloc[UInt64](MT_N)
        self.idx = copy.idx
        for i in range(MT_N):
            self.state.store(i, copy.state[i])

    def __del__(deinit self):
        self.state.free()

    def __getitem__(self, i: Int) -> UInt64:
        return self.state[i]

    def __setitem__(mut self, i: Int, value: UInt64):
        self.state.store(i, value)

    def __len__(self) -> Int:
        return MT_N


# ===----------------------------------------------------------------------=== #
# MT19937 algorithm struct
# ===----------------------------------------------------------------------=== #


struct MT19937(RNGAlgorithm):
    """Mersenne Twister MT19937 random number generator algorithm."""

    var _state: MTState

    def __init__(out self, s: UInt64 = 5489):
        self._state = MTState()
        self.seed(s)

    def __init__(out self, *, deinit take: Self):
        self._state = take._state^

    def __init__(out self, *, copy: Self):
        self._state = copy._state.copy()

    def seed(mut self, s: UInt64):
        """Re-seed the generator."""
        self._state[0] = s
        for i in range(1, MT_N):
            var prev = self._state.state[i - 1]
            self._state.state.store(
                i,
                (1812433253 * (prev ^ (prev >> 30)) + UInt64(i))
                % UInt64(0x100000000),
            )
        self._state.idx = MT_N

    def next(mut self) -> UInt64:
        """Generate next raw 64-bit value."""
        if self._state.idx >= MT_N:
            var i: Int = 0
            while i < MT_N - MT_M:
                var y = (self._state.state[i] & MT_UPPER_MASK) | (
                    self._state.state[i + 1] & MT_LOWER_MASK
                )
                var idx = i + MT_M
                self._state.state.store(
                    i, self._state.state[idx] ^ (y >> 1) ^ MT_MATRIX_A * (y & 1)
                )
                i += 1

            while i < MT_N - 1:
                var y = (self._state.state[i] & MT_UPPER_MASK) | (
                    self._state.state[i + 1] & MT_LOWER_MASK
                )
                var idx = i + MT_M - MT_N
                self._state.state.store(
                    i, self._state.state[idx] ^ (y >> 1) ^ MT_MATRIX_A * (y & 1)
                )
                i += 1

            var y = (self._state.state[MT_N - 1] & MT_UPPER_MASK) | (
                self._state.state[0] & MT_LOWER_MASK
            )
            self._state.state.store(
                MT_N - 1, self._state.state[MT_M - 1] ^ (y >> 1) ^ MT_MATRIX_A * (y & 1)
            )
            self._state.idx = 0

        var y2 = self._state.state[self._state.idx]
        self._state.idx += 1

        y2 ^= y2 >> 11
        y2 ^= (y2 << 7) & 0x9D2C5680
        y2 ^= (y2 << 15) & 0xEFC60000
        y2 ^= y2 >> 18

        return y2


# ===----------------------------------------------------------------------=== #
# RNG parametric wrapper
# ===----------------------------------------------------------------------=== #


struct RNG[T: RNGAlgorithm](Copyable, Movable):
    """Parametric random number generator.

    Wraps any `RNGAlgorithm` implementation and provides the standard
    uniform/integer sampling API. Default algorithm is MT19937.

    Example:
        var rng = RNG[MT19937](seed=42)
        var x = rng.uniform()
    """

    var _algo: Self.T
    var seed_val: UInt64

    def __init__(out self, var algo: Self.T, seed: UInt64):
        self.seed_val = seed
        self._algo = algo^

    def __init__(out self, *, deinit take: Self):
        self.seed_val = take.seed_val
        self._algo = take._algo^

    def __init__(out self, *, copy: Self):
        self.seed_val = copy.seed_val
        self._algo = copy._algo.copy()

    def set_seed(mut self, s: UInt64):
        """Re-seed the generator."""
        self.seed_val = s
        self._algo.seed(s)

    def get(mut self) -> UInt64:
        """Return next raw 64-bit integer."""
        return self._algo.next()

    def uniform(mut self) -> Float64:
        """Return uniform random double in [0, 1)."""
        return Float64(self._algo.next()) / Float64(0x100000000)

    def uniform_pos(mut self) -> Float64:
        """Return uniform random double in (0, 1)."""
        var x = self.uniform()
        while x == 0.0:
            x = self.uniform()
        return x

    def uniform_int(mut self, n: Int) -> Int:
        """Return uniform random integer in [0, n)."""
        return Int(self.uniform() * Float64(n))
