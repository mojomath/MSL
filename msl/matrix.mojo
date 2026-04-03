# SPDX-License-Identifier: GPL-3.0-or-later

# ===----------------------------------------------------------------------=== #
# MSL (Mojo Scientific Library)
#
# Derived from GNU Scientific Library (GSL)
# Original file: gsl_matrix_double.h
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
Matrix for 2D arrays.

A matrix is a 2D array, the basic data structure for linear algebra operations.
"""

from std.memory import UnsafePointer, memset_zero
from std.algorithm.functional import vectorize
from std.sys.info import simd_width_of

from msl.block import Block
from msl.const import MSL_DBL_EPSILON
from msl.types import STATUS_SUCCESS, STATUS_FAILURE


# ===----------------------------------------------------------------------=== #
# Matrix struct
# ===----------------------------------------------------------------------=== #

@explicit_destroy("Must call .free()")
struct Matrix(Copyable, Movable):
    """GSL matrix structure for 2D arrays."""

    var s1: Int
    """Number of rows. (size1 in gsl)."""
    var s2: Int
    """Number of columns. (size2 in gsl)."""
    var tda: Int
    var data: UnsafePointer[Float64, MutExt]
    var block: UnsafePointer[Block, MutExt]
    var owner: Int

    def __init__(out self, n1: Int, n2: Int, *, initialize: Bool = False):
        self.s1 = n1
        self.s2 = n2
        self.tda = n2
        var blk = Block(n1 * n2, initialize=initialize)
        self.data = blk.data
        self.block = alloc[Block](1)
        self.block.init_pointee_move(blk^)
        self.owner = 1

    def free(deinit self):
        if self.owner == 1 and self.block != UnsafePointer[Block, MutExt]():
            var blk = self.block.take_pointee()
            blk^.free()
            self.block.free()

    def size1(self) -> Int:
        """Return number of rows."""
        return self.s1

    def size2(self) -> Int:
        """Return number of columns."""
        return self.s2

    def ptr_read(ref self) -> UnsafePointer[Float64, origin_of(self)]:
        """Return pointer to the matrix data."""
        return rebind[UnsafePointer[Float64, origin_of(self)]](self.data.unsafe_mut_cast[False]().unsafe_origin_cast[origin_of(self)]())

    def ptr_mut(mut self) -> UnsafePointer[Float64, origin_of(self)]:
        """Return mutable pointer to the matrix data."""
        return self.data.unsafe_origin_cast[origin_of(self)]()

    def get(self, i: Int, j: Int) -> Float64:
        """Get element at row i, column j."""
        return self.data[i * self.tda + j]

    def set(self, i: Int, j: Int, value: Float64):
        """Set element at row i, column j to value."""
        self.data[i * self.tda + j] = value

    def __getitem__(self, i: Int, j: Int) -> Float64:
        """Get element at row i, column j."""
        return self.get(i, j)

    def __setitem__(self, i: Int, j: Int, value: Float64):
        """Set element at row i, column j."""
        self.set(i, j, value)

    @always_inline
    def nelems(self) -> Int:
        """Return total number of elements."""
        return self.s1 * self.s2

    @staticmethod
    def zero(n1: Int, n2: Int) -> Matrix:
        """Return a zero-initialized matrix."""
        return Matrix(n1, n2, initialize=True)

    @staticmethod
    def identity(n: Int) -> Matrix:
        """Return an identity matrix of size n x n."""
        var mat = Matrix(n, n, initialize=True)
        for i in range(n):
            mat.data[i * mat.tda + i] = 1.0
        return mat^

    @staticmethod
    def all(n1: Int, n2: Int, x: Float64) -> Matrix:
        """Return a matrix of size n1 x n2 with all elements set to x."""
        var mat = Matrix(n1, n2, initialize=True)
        comptime simd_width: Int = simd_width_of[f64]()
        @parameter
        def closure[width: Int](i: Int) unified {mut mat, read x}:
            mat.data[i] = x
        vectorize[simd_width](mat.nelems(), closure)
        return mat^


# ===----------------------------------------------------------------------=== #
# Matrix operations
# ===----------------------------------------------------------------------=== #

def matrix_alloc(n1: Int, n2: Int) -> Matrix:
    """Allocate a matrix of n1 x n2 doubles.

    Args:
        n1: Number of rows.
        n2: Number of columns.

    Returns:
        Matrix with allocated data.
    """
    return Matrix(n1, n2)


def matrix_calloc(n1: Int, n2: Int) -> Matrix:
    """Allocate and zero-initialize a matrix of n1 x n2 doubles.

    Args:
        n1: Number of rows.
        n2: Number of columns.

    Returns:
        Matrix with allocated and zeroed data.
    """
    return Matrix(n1, n2, initialize=True)


def matrix_size1(mat: Matrix) -> Int:
    """Return number of rows.

    Args:
        mat: Matrix to query.

    Returns:
        Number of rows.
    """
    return mat.s1


def matrix_size2(mat: Matrix) -> Int:
    """Return number of columns.

    Args:
        mat: Matrix to query.

    Returns:
        Number of columns.
    """
    return mat.s2


def matrix_set_zero(mut mat: Matrix):
    """Set all elements to zero.

    Args:
        mat: Matrix to zero.
    """
    memset_zero(mat.ptr_mut(), mat.s1 * mat.s2)


def matrix_set_all(mut mat: Matrix, x: Float64):
    """Set all elements to x.

    Args:
        mat: Matrix to set.
        x: Value to set.
    """
    comptime simd_width: Int = simd_width_of[f64]()
    @parameter
    def closure[width: Int](i: Int) unified {mut mat, read x}:
        mat.data[i] = x
    vectorize[simd_width](mat.nelems(), closure)


def matrix_set_identity(mut mat: Matrix):
    """Set matrix to identity (1 on diagonal, 0 elsewhere).

    Args:
        mat: Matrix to set.
    """
    matrix_set_zero(mat)
    for i in range(mat.s1):
        if i < mat.s2:
            mat.data[i * mat.tda + i] = 1.0
