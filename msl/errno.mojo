# SPDX-License-Identifier: GPL-3.0-or-later

# ===----------------------------------------------------------------------=== #
# MSL (Mojo Scientific Library)
#
# Derived from GNU Scientific Library (GSL)
# Original file: block/block_source.c
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
Error codes and error handling.

Pure Mojo implementation of GSL error handling.
"""

from std.ffi import c_int
from std.memory import UnsafePointer


# ===----------------------------------------------------------------------=== #
# Error codes (from gsl_errno.h)
# ===----------------------------------------------------------------------=== #

comptime GSL_SUCCESS: Int = 0
"""Success."""

comptime GSL_FAILURE: Int = -1
"""Generic failure."""

comptime GSL_CONTINUE: Int = -2
"""Iteration has not converged."""

comptime GSL_EDOM: Int = 1
"""Input domain error, e.g. sqrt(-1)."""

comptime GSL_ERANGE: Int = 2
"""Output range error, e.g. exp(1e100)."""

comptime GSL_EFAULT: Int = 3
"""Invalid pointer."""

comptime GSL_EINVAL: Int = 4
"""Invalid argument."""

comptime GSL_EFAILED: Int = 5
"""Generic failure."""

comptime GSL_EFACTOR: Int = 6
"""Factorization failed."""

comptime GSL_ESANITY: Int = 7
"""Sanity check failed."""

comptime GSL_ENOMEM: Int = 8
"""malloc failed."""

comptime GSL_EBADFUNC: Int = 9
"""Problem with user-supplied function."""

comptime GSL_ERUNAWAY: Int = 10
"""Iterative process is out of control."""

comptime GSL_EMAXITER: Int = 11
"""Exceeded max number of iterations."""

comptime GSL_EZERODIV: Int = 12
"""Tried to divide by zero."""

comptime GSL_EBADTOL: Int = 13
"""User specified an invalid tolerance."""

comptime GSL_ETOL: Int = 14
"""Failed to reach specified tolerance."""

comptime GSL_EUNDRFLW: Int = 15
"""Underflow."""

comptime GSL_EOVRFLW: Int = 16
"""Overflow."""

comptime GSL_ELOSS: Int = 17
"""Loss of accuracy."""

comptime GSL_EROUND: Int = 18
"""Failed because of roundoff error."""

comptime GSL_EBADLEN: Int = 19
"""Matrix/vector lengths not conformant."""

comptime GSL_ENOTSQR: Int = 20
"""Matrix not square."""

comptime GSL_ESING: Int = 21
"""Apparent singularity detected."""

comptime GSL_EDIVERGE: Int = 22
"""Integral or series divergent."""

comptime GSL_EUNSUP: Int = 23
"""Requested feature not supported."""

comptime GSL_EUNIMPL: Int = 24
"""Requested feature not implemented."""

comptime GSL_ECACHE: Int = 25
"""Cache limit exceeded."""

comptime GSL_ETABLE: Int = 26
"""Table limit exceeded."""

comptime GSL_ENOPROG: Int = 27
"""Iteration not making progress."""

comptime GSL_ENOPROGJ: Int = 28
"""Jacobian not improving solution."""

comptime GSL_ETOLF: Int = 29
"""Cannot reach tolerance in F."""

comptime GSL_ETOLX: Int = 30
"""Cannot reach tolerance in X."""

comptime GSL_ETOLG: Int = 31
"""Cannot reach tolerance in gradient."""

comptime GSL_EOF: Int = 32
"""End of file."""


# ===----------------------------------------------------------------------=== #
# Error handler type
# ===----------------------------------------------------------------------=== #

# comptime gsl_error_handler_t = def [NoneType, UnsafePointer[c_char], Int, Int]()
# """Function pointer type for error handler."""


# ===----------------------------------------------------------------------=== #
# Error reporting
# ===----------------------------------------------------------------------=== #


def msl_error(reason: String, file: String, line: Int, errno: Int):
    """Report an error."""
    print(reason + " in " + file + " line " + str(line))
    print("Error code: " + str(errno))


def msl_assert(reason: String, file: String, line: Int):
    """Assert an error."""
    print("Assertion failure: " + reason + " in " + file + " line " + str(line))
