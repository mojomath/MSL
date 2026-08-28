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
"""

from std.ffi import c_int
from std.memory import Pointer


# ===----------------------------------------------------------------------=== #
# Error codes
# ===----------------------------------------------------------------------=== #

comptime MSL_SUCCESS: Int = 0
"""Success."""

comptime MSL_FAILURE: Int = -1
"""Generic failure."""

comptime MSL_CONTINUE: Int = -2
"""Iteration has not converged."""

comptime MSL_EDOM: Int = 1
"""Input domain error, e.g. sqrt(-1)."""

comptime MSL_ERANGE: Int = 2
"""Output range error, e.g. exp(1e100)."""

comptime MSL_EFAULT: Int = 3
"""Invalid pointer."""

comptime MSL_EINVAL: Int = 4
"""Invalid argument."""

comptime MSL_EFAILED: Int = 5
"""Generic failure."""

comptime MSL_EFACTOR: Int = 6
"""Factorization failed."""

comptime MSL_ESANITY: Int = 7
"""Sanity check failed."""

comptime MSL_ENOMEM: Int = 8
"""Malloc failed."""

comptime MSL_EBADFUNC: Int = 9
"""Problem with user-supplied function."""

comptime MSL_ERUNAWAY: Int = 10
"""Iterative process is out of control."""

comptime MSL_EMAXITER: Int = 11
"""Exceeded max number of iterations."""

comptime MSL_EZERODIV: Int = 12
"""Tried to divide by zero."""

comptime MSL_EBADTOL: Int = 13
"""User specified an invalid tolerance."""

comptime MSL_ETOL: Int = 14
"""Failed to reach specified tolerance."""

comptime MSL_EUNDRFLW: Int = 15
"""Underflow."""

comptime MSL_EOVRFLW: Int = 16
"""Overflow."""

comptime MSL_ELOSS: Int = 17
"""Loss of accuracy."""

comptime MSL_EROUND: Int = 18
"""Failed because of roundoff error."""

comptime MSL_EBADLEN: Int = 19
"""Matrix/vector lengths not conformant."""

comptime MSL_ENOTSQR: Int = 20
"""Matrix not square."""

comptime MSL_ESING: Int = 21
"""Apparent singularity detected."""

comptime MSL_EDIVERGE: Int = 22
"""Integral or series divergent."""

comptime MSL_EUNSUP: Int = 23
"""Requested feature not supported."""

comptime MSL_EUNIMPL: Int = 24
"""Requested feature not implemented."""

comptime MSL_ECACHE: Int = 25
"""Cache limit exceeded."""

comptime MSL_ETABLE: Int = 26
"""Table limit exceeded."""

comptime MSL_ENOPROG: Int = 27
"""Iteration not making progress."""

comptime MSL_ENOPROGJ: Int = 28
"""Jacobian not improving solution."""

comptime MSL_ETOLF: Int = 29
"""Cannot reach tolerance in F."""

comptime MSL_ETOLX: Int = 30
"""Cannot reach tolerance in X."""

comptime MSL_ETOLG: Int = 31
"""Cannot reach tolerance in gradient."""

comptime MSL_EOF: Int = 32
"""End of file."""


# ===----------------------------------------------------------------------=== #
# Error reporting
# ===----------------------------------------------------------------------=== #


def msl_error(reason: String, file: String, line: Int, errno: Int) raises:
    """Report an error by raising."""
    raise Error(
        reason + " (code " + String(errno) + ") in " + file + ":" + String(line)
    )


def msl_assert(reason: String, file: String, line: Int) raises:
    """Assert an error by raising."""
    raise Error(
        "Assertion failure: " + reason + " in " + file + ":" + String(line)
    )
