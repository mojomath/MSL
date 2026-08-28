# SPDX-License-Identifier: GPL-3.0-or-later

# ===----------------------------------------------------------------------=== #
# MSL (Mojo Scientific Library)
#
# Derived from GNU Scientific Library (GSL)
# Original file: integration/gsl_integration.h
#
# Original authors:
# Copyright (C) 1996-2007 Brian Gough
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
Workspace and result types for numerical integration.
"""

from std.memory import Pointer, unsafe_memset_zero
from std.memory.alloc import unsafe_alloc

from msl.core.const import MSL_DBL_MAX

comptime MutExt = MutUntrackedOrigin

comptime MSL_KRONROD_15: Int = 1
comptime MSL_KRONROD_21: Int = 2
comptime MSL_KRONROD_31: Int = 3
comptime MSL_KRONROD_41: Int = 4
comptime MSL_KRONROD_51: Int = 5
comptime MSL_KRONROD_61: Int = 6


struct QKResult(Copyable, Movable):
    var result: Float64
    var abserr: Float64
    var resabs: Float64
    var resasc: Float64

    def __init__(out self):
        self.result = 0.0
        self.abserr = 0.0
        self.resabs = 0.0
        self.resasc = 0.0

    def __init__(out self, *, copy: Self):
        self.result = copy.result
        self.abserr = copy.abserr
        self.resabs = copy.resabs
        self.resasc = copy.resasc

    def __init__(out self, *, deinit move: Self):
        self.result = move.result
        self.abserr = move.abserr
        self.resabs = move.resabs
        self.resasc = move.resasc


struct IntegrationResult(Copyable, Movable):
    """Result of an integration operation."""

    var val: Float64
    var err: Float64

    def __init__(out self):
        self.val = 0.0
        self.err = 0.0

    def __init__(out self, *, copy: Self):
        self.val = copy.val
        self.err = copy.err

    def __init__(out self, *, deinit move: Self):
        self.val = move.val
        self.err = move.err


struct IntegrationWorkspace(Movable):
    """Adaptive integration workspace storing the interval heap.

    Holds up to `limit` subintervals, sorted by descending error.
    """

    var limit: Int
    var size: Int
    var nrmax: Int
    var i: Int
    var maximum_level: Int
    var alist: Pointer[Float64, MutExt]
    var blist: Pointer[Float64, MutExt]
    var rlist: Pointer[Float64, MutExt]
    var elist: Pointer[Float64, MutExt]
    var order: Pointer[Int, MutExt]
    var level: Pointer[Int, MutExt]

    def __init__(out self, limit: Int):
        self.limit = limit
        self.size = 0
        self.nrmax = 0
        self.i = 0
        self.maximum_level = 0
        self.alist = unsafe_alloc[Float64](limit)
        self.blist = unsafe_alloc[Float64](limit)
        self.rlist = unsafe_alloc[Float64](limit)
        self.elist = unsafe_alloc[Float64](limit)
        self.order = unsafe_alloc[Int](limit)
        self.level = unsafe_alloc[Int](limit)
        unsafe_memset_zero(self.alist, limit)
        unsafe_memset_zero(self.blist, limit)
        unsafe_memset_zero(self.rlist, limit)
        unsafe_memset_zero(self.elist, limit)
        unsafe_memset_zero(self.order, limit)
        unsafe_memset_zero(self.level, limit)

    def __init__(out self, *, deinit move: Self):
        self.limit = move.limit
        self.size = move.size
        self.nrmax = move.nrmax
        self.i = move.i
        self.maximum_level = move.maximum_level
        self.alist = move.alist
        self.blist = move.blist
        self.rlist = move.rlist
        self.elist = move.elist
        self.order = move.order
        self.level = move.level

    def __del__(deinit self):
        self.alist.unsafe_free()
        self.blist.unsafe_free()
        self.rlist.unsafe_free()
        self.elist.unsafe_free()
        self.order.unsafe_free()
        self.level.unsafe_free()

    def initialise(
        mut self, a: Float64, b: Float64, result: Float64, error: Float64
    ):
        """Set up workspace with the initial interval [a,b]."""
        self.size = 1
        self.nrmax = 0
        self.i = 0
        self.maximum_level = 0
        self.alist[0] = a
        self.blist[0] = b
        self.rlist[0] = result
        self.elist[0] = error
        self.order[0] = 0
        self.level[0] = 0

    def retrieve(self) -> Tuple[Float64, Float64, Float64, Float64]:
        """Return (a, b, result, error) for the current worst interval."""
        return (
            self.alist[self.i],
            self.blist[self.i],
            self.rlist[self.i],
            self.elist[self.i],
        )

    def update(
        mut self,
        a1: Float64,
        b1: Float64,
        area1: Float64,
        error1: Float64,
        a2: Float64,
        b2: Float64,
        area2: Float64,
        error2: Float64,
    ):
        """Store the two children of the bisected interval and run qpsrt."""
        var i_max = self.i
        var i_new = self.size
        var new_level = self.level[i_max] + 1

        if error2 > error1:
            self.alist[i_max] = a2
            self.blist[i_max] = b2
            self.rlist[i_max] = area2
            self.elist[i_max] = error2
            self.level[i_max] = new_level

            self.alist[i_new] = a1
            self.blist[i_new] = b1
            self.rlist[i_new] = area1
            self.elist[i_new] = error1
            self.level[i_new] = new_level
        else:
            self.alist[i_max] = a1
            self.blist[i_max] = b1
            self.rlist[i_max] = area1
            self.elist[i_max] = error1
            self.level[i_max] = new_level

            self.alist[i_new] = a2
            self.blist[i_new] = b2
            self.rlist[i_new] = area2
            self.elist[i_new] = error2
            self.level[i_new] = new_level

        self.size += 1
        if new_level > self.maximum_level:
            self.maximum_level = new_level

        self._qpsrt()

    def _qpsrt(mut self):
        """Maintain order[] as descending-by-error index array."""
        var last = self.size - 1
        var i_nrmax = self.nrmax
        var i_maxerr = self.order[i_nrmax]

        if last < 2:
            self.order[0] = 0
            self.order[1] = 1
            self.i = i_maxerr
            return

        var errmax = self.elist[i_maxerr]

        while i_nrmax > 0 and errmax > self.elist[self.order[i_nrmax - 1]]:
            self.order[i_nrmax] = self.order[i_nrmax - 1]
            i_nrmax -= 1

        var top: Int
        if last < self.limit // 2 + 2:
            top = last
        else:
            top = self.limit - last + 1

        var idx = i_nrmax + 1
        while idx < top and errmax < self.elist[self.order[idx]]:
            self.order[idx - 1] = self.order[idx]
            idx += 1
        self.order[idx - 1] = i_maxerr

        var errmin = self.elist[last]
        var k = top - 1
        while k > idx - 2 and errmin >= self.elist[self.order[k]]:
            self.order[k + 1] = self.order[k]
            k -= 1
        self.order[k + 1] = last

        self.i = self.order[i_nrmax]
        self.nrmax = i_nrmax

    def sum_results(self) -> Float64:
        """Sum all rlist entries."""
        var total: Float64 = 0.0
        for k in range(self.size):
            total += self.rlist[k]
        return total

    def large_interval(self) -> Bool:
        return self.level[self.i] < self.maximum_level

    def increase_nrmax(mut self) -> Bool:
        """Advance nrmax past fully-refined intervals. Returns True if advanced.
        """
        var id = self.nrmax
        var last = self.size - 1
        var jupbnd = (
            last if last > (1 + self.limit // 2) else self.limit + 1 - last
        )
        for _ in range(jupbnd - id):
            var i_max = self.order[self.nrmax]
            self.i = i_max
            if self.level[i_max] < self.maximum_level:
                return True
            self.nrmax += 1
        return False

    def reset_nrmax(mut self):
        self.nrmax = 0
        self.i = self.order[0]
