# SPDX-License-Identifier: GPL-3.0-or-later

# ===----------------------------------------------------------------------=== #
# MSL (Mojo Scientific Library)
#
# Derived from GNU Scientific Library (GSL)
# Original file: gsl_vector_double.h
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
Vector for 1D arrays with stride.

A vector is a 1D array with stride, the basic data structure for
linear algebra operations.
"""

from std.memory import UnsafePointer, memset_zero
from std.algorithm.functional import vectorize
from std.sys.info import simd_width_of

from msl.core import Block
from msl.core.const import MSL_DBL_EPSILON
from msl.core.types import STATUS_SUCCESS, STATUS_FAILURE


# ===----------------------------------------------------------------------=== #
# Vector struct
# ===----------------------------------------------------------------------=== #


# NOTE: I have kept the fields to be same as GSL for compatibility, but we will move to using matmojo so this is only temporary.
struct Vector(Copyable, Movable):
    """GSL vector structure for 1D arrays with stride."""

    var size: Int
    var stride: Int
    var data: UnsafePointer[Float64, MutExt]
    var block: Optional[UnsafePointer[Block, MutExt]]
    var owner: Int

    def __init__(out self, size: Int, *, initialize: Bool = False):
        self.size = size
        self.stride = 1
        var blk = Block(size, initialize=initialize)
        self.data = blk.data
        var bptr = alloc[Block](1)
        bptr.init_pointee_move(blk^)
        self.block = bptr
        self.owner = 1

    def __init__(out self, size: Int, stride: Int, *, initialize: Bool = False):
        self.size = size
        self.stride = stride
        var blk = Block(size * stride, initialize=initialize)
        self.data = blk.data
        var bptr = alloc[Block](1)
        bptr.init_pointee_move(blk^)
        self.block = bptr
        self.owner = 1

    def __del__(deinit self):
        if self.owner == 1 and self.block:
            self.block.value().destroy_pointee()
            self.block.value().free()

    def get_size(self) -> Int:
        """Return the size of the vector."""
        return self.size

    def get_stride(self) -> Int:
        """Return the stride of the vector."""
        return self.stride

    def ptr_mut(mut self) -> UnsafePointer[Float64, origin_of(self)]:
        """Return pointer to the vector data."""
        return self.data.unsafe_origin_cast[origin_of(self)]()

    def ptr_read(ref self) -> UnsafePointer[Float64, origin_of(self)]:
        """Return pointer to the vector data."""
        return rebind[UnsafePointer[Float64, origin_of(self)]](
            self.data.unsafe_mut_cast[False]().unsafe_origin_cast[
                origin_of(self)
            ]()
        )

    def get(self, i: Int) -> Float64:
        """Get element at index i."""
        return self.data[i * self.stride]

    def set(self, i: Int, value: Float64):
        """Set element at index i to value."""
        self.data[i * self.stride] = value

    def __getitem__(self, i: Int) -> Float64:
        """Get element at index i."""
        return self.get(i)

    def __setitem__(self, i: Int, value: Float64):
        """Set element at index i."""
        self.set(i, value)


# ===----------------------------------------------------------------------=== #
# Vector operations
# ===----------------------------------------------------------------------=== #


def vector_alloc(n: Int) -> Vector:
    """Allocate a vector of n doubles.

    Args:
        n: Number of elements.

    Returns:
        Vector with allocated data.
    """
    return Vector(n)


def vector_calloc(n: Int) -> Vector:
    """Allocate and zero-initialize a vector of n doubles.

    Args:
        n: Number of elements.

    Returns:
        Vector with allocated and zeroed data.
    """
    return Vector(n, initialize=True)


def vector_size(vec: Vector) -> Int:
    """Return the size of the vector.

    Args:
        vec: Vector to query.

    Returns:
        Number of elements in the vector.
    """
    return vec.size


def vector_stride(vec: Vector) -> Int:
    """Return the stride of the vector.

    Args:
        vec: Vector to query.

    Returns:
        Stride of the vector.
    """
    return vec.stride


def vector_set_zero(vec: Vector):
    """Set all elements to zero.

    Args:
        vec: Vector to zero.
    """
    memset_zero(vec.data, vec.size * vec.stride)


def vector_set_all(mut vec: Vector, x: Float64):
    """Set all elements to x.

    Args:
        vec: Vector to set.
        x: Value to set.
    """
    comptime simd_width: Int = simd_width_of[f64]()

    def closure[width: Int](i: Int) {mut vec, x}:
        vec.data[i * vec.stride] = x

    vectorize[simd_width](vec.size, closure)
