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

from std.memory import Pointer, unsafe_memset_zero
from std.algorithm.functional import vectorize
from std.sys.info import simd_width_of
from std.math import sqrt

from msl.core.const import MSL_DBL_EPSILON


# ===----------------------------------------------------------------------=== #
# Vector struct
# ===----------------------------------------------------------------------=== #


struct Vector(Copyable, Movable):
    """1D array with stride."""

    var size: Int
    var stride: Int
    var data: Pointer[Float64, MutExt]
    var owner: Bool

    def __init__(out self, size: Int, *, initialize: Bool = False):
        self.size = size
        self.stride = 1
        self.data = alloc[Float64](size)
        if initialize:
            unsafe_memset_zero(self.data, size)
        self.owner = True

    def __init__(out self, size: Int, stride: Int, *, initialize: Bool = False):
        self.size = size
        self.stride = stride
        self.data = alloc[Float64](size * stride)
        if initialize:
            unsafe_memset_zero(self.data, size * stride)
        self.owner = True

    def __init__[
        origin: Origin, //
    ](
        out self,
        ptr: Pointer[Float64, origin],
        size: Int,
        stride: Int = 1,
    ):
        """Create a non-owning view over an externally managed buffer.

        The caller is responsible for keeping the buffer alive for the
        lifetime of this Vector. No memory is allocated or freed.

        Args:
            ptr: Pointer to the first element of the buffer.
            size: Number of logical elements.
            stride: Element stride (default 1 = contiguous).
        """
        self.size = size
        self.stride = stride
        self.data = rebind[Pointer[Float64, MutExt]](ptr)
        self.owner = False

    def __del__(deinit self):
        if self.owner:
            self.data.free()

    def get_size(self) -> Int:
        """Return the size of the vector."""
        return self.size

    def get_stride(self) -> Int:
        """Return the stride of the vector."""
        return self.stride

    def mut_ptr(mut self) -> Pointer[Float64, origin_of(self)]:
        """Return pointer to the vector data."""
        return self.data.unsafe_origin_cast[origin_of(self)]()

    def immut_ptr(ref self) -> Pointer[Float64, origin_of(self)]:
        """Return pointer to the vector data."""
        return rebind[Pointer[Float64, origin_of(self)]](
            self.data.unsafe_mut_cast[False]().unsafe_origin_cast[
                origin_of(self)
            ]()
        )

    def get(self, i: Int) -> Float64:
        """Get element at index i."""
        return self.data[unsafe_offset=i * self.stride]

    def set(self, i: Int, value: Float64):
        """Set element at index i to value."""
        self.data[unsafe_offset=i * self.stride] = value

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
    unsafe_memset_zero(vec.data, vec.size * vec.stride)


def vector_set_all(mut vec: Vector, x: Float64):
    """Set all elements to x.

    Args:
        vec: Vector to set.
        x: Value to set.
    """
    comptime simd_width: Int = simd_width_of[DType.float64]()

    def closure[width: Int](i: Int) {mut vec, x}:
        vec.data[unsafe_offset=i * vec.stride] = x

    vectorize[simd_width](vec.size, closure)


def vector_add(mut a: Vector, b: Vector):
    """Add vector b into a in-place: a[i] += b[i].

    Args:
        a: Destination vector (mutated).
        b: Source vector.
    """
    for i in range(a.size):
        a.data[unsafe_offset=i * a.stride] += b.data[unsafe_offset=i * b.stride]


def vector_sub(mut a: Vector, b: Vector):
    """Subtract vector b from a in-place: a[i] -= b[i].

    Args:
        a: Destination vector (mutated).
        b: Source vector.
    """
    for i in range(a.size):
        a.data[unsafe_offset=i * a.stride] -= b.data[unsafe_offset=i * b.stride]


def vector_scale(mut vec: Vector, alpha: Float64):
    """Scale all elements by alpha in-place: vec[i] *= alpha.

    Args:
        vec: Vector to scale.
        alpha: Scalar multiplier.
    """
    comptime simd_width: Int = simd_width_of[DType.float64]()

    def closure[width: Int](i: Int) {mut vec, alpha}:
        vec.data[unsafe_offset=i * vec.stride] *= alpha

    vectorize[simd_width](vec.size, closure)


def vector_axpy(alpha: Float64, x: Vector, mut y: Vector):
    """AXPY: y[i] += alpha * x[i].

    Args:
        alpha: Scalar multiplier.
        x: Source vector.
        y: Destination vector (mutated).
    """
    for i in range(y.size):
        y.data[unsafe_offset=i * y.stride] += alpha * x.data[unsafe_offset=i * x.stride]


def vector_dot(a: Vector, b: Vector) -> Float64:
    """Dot product: sum of a[i] * b[i].

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Dot product value.
    """
    var result: Float64 = 0.0
    for i in range(a.size):
        result += a.data[unsafe_offset=i * a.stride] * b.data[unsafe_offset=i * b.stride]
    return result


def vector_norm(vec: Vector) -> Float64:
    """Euclidean norm: sqrt(sum of vec[i]^2).

    Args:
        vec: Vector to compute norm of.

    Returns:
        Euclidean norm.
    """
    var result: Float64 = 0.0
    for i in range(vec.size):
        var v = vec.data[unsafe_offset=i * vec.stride]
        result += v * v
    return sqrt(result)
