# SPDX-License-Identifier: GPL-3.0-or-later

# ===----------------------------------------------------------------------=== #
# MSL (Mojo Scientific Library)
#
# Derived from GNU Scientific Library (GSL)
# Original file: gsl_permutation.h
#
# Original authors:
# Copyright (C) 1996–2007 Gerard Jungman, Brian Gough
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
Permutation operations.

Pure Mojo implementation of permutation operations for reordering
elements in vectors and matrices.
"""

from std.memory import Pointer
from msl import MutExt


# ===----------------------------------------------------------------------=== #
# Permutation struct
# ===----------------------------------------------------------------------=== #


struct Permutation(Copyable, Movable):
    """Permutation array for reordering elements."""

    # TODO: change to Int once mojo align them both.
    var data: Pointer[Scalar[DType.int], MutExt]
    var size: Int

    def __init__(out self, n: Int):
        self.size = n
        self.data = unsafe_alloc[Scalar[DType.int]](n)
        for i in range(n):
            self.data.unsafe_store(i, Scalar[DType.int](i))

    def __del__(deinit self):
        self.data.unsafe_free()

    def get(self, i: Int) -> Int:
        """Get element at index i."""
        return Int(self.data[unsafe_offset=i])

    def set(self, i: Int, value: Int):
        """Set element at index i."""
        self.data.unsafe_store(i, Scalar[DType.int](value))

    def swap(self, i: Int, j: Int):
        """Swap elements at indices i and j."""
        var temp = self.data[unsafe_offset=i]
        self.data.unsafe_store(i, self.data[unsafe_offset=j])
        self.data.unsafe_store(j, temp)

    def inversions(self) -> Int:
        """Count the number of inversions."""
        var inv: Int = 0
        for i in range(self.size):
            for j in range(i + 1, self.size):
                if self.data[unsafe_offset=i] > self.data[unsafe_offset=j]:
                    inv += 1
        return inv


# ===----------------------------------------------------------------------=== #
# Permutation functions
# ===----------------------------------------------------------------------=== #


def permutation_alloc(n: Int) -> Permutation:
    return Permutation(n)


def permutation_init(p: Permutation):
    for i in range(p.size):
        p.data.unsafe_store(i, Scalar[DType.int](i))


def permutation_next(p: Permutation) -> Bool:
    if p.size <= 1:
        return False

    var i = p.size - 2
    while i >= 0:
        if p.data[unsafe_offset=i] < p.data[unsafe_offset=i + 1]:
            break
        i -= 1

    if i < 0:
        return False

    var j = p.size - 1
    while j > i:
        if p.data[unsafe_offset=j] > p.data[unsafe_offset=i]:
            break
        j -= 1

    var temp = p.data[unsafe_offset=i]
    p.data.unsafe_store(i, p.data[unsafe_offset=j])
    p.data.unsafe_store(j, temp)

    var left = i + 1
    var right = p.size - 1
    while left < right:
        var t = p.data[unsafe_offset=left]
        p.data.unsafe_store(left, p.data[unsafe_offset=right])
        p.data.unsafe_store(right, t)
        left += 1
        right -= 1

    return True
