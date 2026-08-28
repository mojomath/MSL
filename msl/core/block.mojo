# SPDX-License-Identifier: GPL-3.0-or-later
# MSL (Mojo Scientific Library)
#
# Derived from GNU Scientific Library (GSL)
# Original file: gsl_block_double.h
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
Block for contiguous array storage.

A block is a simple array wrapper that manages memory allocation
for contiguous data storage. Vectors can be created from blocks.
"""

# ===----------------------------------------------------------------------=== #
# Stdlib
# ===----------------------------------------------------------------------=== #
from std.memory import (
    Pointer,
    unsafe_memcpy,
    unsafe_memset_zero,
)
from std.memory.alloc import unsafe_alloc

from msl import MutExt

# ===----------------------------------------------------------------------=== #
# Block struct
# ===----------------------------------------------------------------------=== #
# IDEA: Perhaps it's better to use NuMojo DataContainer directly here since we want to have zero-copy interop.


struct Block(Copyable, Movable):
    """Contiguous double array block."""

    var size: Int
    var data: Pointer[Float64, MutExt]

    def __init__(out self, size: Int, *, initialize: Bool = False):
        self.size = size
        self.data = unsafe_alloc[Float64](size)
        if initialize:
            unsafe_memset_zero(self.data, size)

    def __init__(out self, *, copy: Self):
        self.size = copy.size
        self.data = unsafe_alloc[Float64](copy.size)
        if copy.size > 0:
            unsafe_memcpy(dest=self.data, src=copy.data, count=copy.size)

    def __init__(out self, *, deinit move: Self):
        self.size = move.size
        self.data = move.data

    def __deinit__(deinit self):
        if self.size > 0:
            self.data.unsafe_free()

    def nelems(self) -> Int:
        """Return the size of the block."""
        return self.size

    def mut_ptr(mut self) -> Pointer[Float64, origin_of(self)]:
        """Return pointer to the block data."""
        return self.data.unsafe_origin_cast[origin_of(self)]()

    def immut_ptr(self) -> Pointer[Float64, origin_of(self)]:
        """Return pointer to the block data."""
        return self.data.as_imm().unsafe_origin_cast[origin_of(self)]()


# ===----------------------------------------------------------------------=== #
# Block operations
# ===----------------------------------------------------------------------=== #
# TODO: These are provided as standalone functions to mirror the C API, even though they could be
# methods on the Block struct in a more idiomatic Mojo design. I might consider removing them
# and use method directly on block, this is more safe because of the mojo's origin system.
# For now, I'll test them and see.


def block_alloc(n: Int) -> Block:
    """Allocate a block of n doubles.

    Args:
        n: Number of elements to allocate.

    Returns:
        Block with allocated data.
    """
    return Block(n)


def block_calloc(n: Int) -> Block:
    """Allocate and zero-initialize a block of n doubles.

    Args:
        n: Number of elements to allocate.

    Returns:
        Block with allocated and zeroed data.
    """
    return Block(n, initialize=True)


def block_size(block: Block) -> Int:
    """Return the size of the block.

    Args:
        block: Block to query.

    Returns:
        Number of elements in the block.
    """
    return block.size


def block_data(mut block: Block) -> Pointer[Float64, origin_of(block)]:
    """Return pointer to the block data.

    Args:
        block: Block to query.

    Returns:
        Pointer to data array.
    """
    return block.data.unsafe_origin_cast[origin_of(block)]()
