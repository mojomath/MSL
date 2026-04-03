# SPDX-License-Identifier: GPL-3.0-or-later

# ===----------------------------------------------------------------------=== #
# MSL (Mojo Scientific Library)
#
# Derived from GNU Scientific Library (GSL)
# Original file: specfunc/gsl_sf_result.h
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
# ===----------------------------------------------------------------------=== #

"""Special function result types."""

from msl.core.const import MSL_DBL_EPSILON


struct SFSResult(ImplicitlyCopyable, Movable, Writable):
    var val: Float64
    var err: Float64

    def __init__(out self, val: Float64, err: Float64):
        self.val = val
        self.err = err

    def __init__(out self):
        self.val = 0.0
        self.err = 0.0

    def write_to[W: Writer](self, mut writer: W):
        writer.write("val: ", self.val, ", err: ", self.err)


struct SFSResultE10:
    var val: Float64
    var err: Float64
    var e10: Int

    def __init__(out self, val: Float64, err: Float64, e10: Int):
        self.val = val
        self.err = err
        self.e10 = e10

    def __init__(out self):
        self.val = 0.0
        self.err = 0.0
        self.e10 = 0

    def write_to[W: Writer](self, mut writer: W):
        writer.write(
            "val: ", self.val, ", err: ", self.err, ", e10: ", self.e10
        )


def sf_result_set(mut result: SFSResult, val: Float64, err: Float64):
    """Set result value and error estimate."""
    result.val = val
    result.err = err


def sf_result_smash_e(mut result: SFSResult, re: SFSResultE10) -> Int:
    """Convert result_e10 to regular result."""
    if re.e10 > 307:
        return 1
    if re.e10 < -307:
        return 1

    var factor = 1.0
    for _ in range(re.e10):
        factor *= 10.0

    result.val = re.val * factor
    result.err = re.err * factor
    return 0
