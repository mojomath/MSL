# SPDX-License-Identifier: GPL-3.0-or-later

# ===----------------------------------------------------------------------=== #
# MSL (Mojo Scientific Library)
#
# Test file for integration module
#
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

from std.testing import TestSuite
from std.math import sin, exp, cos, abs

from msl.integration import qk15, qng_integrate


def tolerance(x: Float64, expected: Float64, tol: Float64) -> Bool:
    return abs(x - expected) < tol

def func_linear(x: Float64) capturing -> Float64:
    return x

def func_quadratic(x: Float64) capturing -> Float64:
    return x * x


def func_sin(x: Float64) capturing -> Float64:
    return sin(x)


def func_exp(x: Float64) capturing -> Float64:
    return exp(x)


def test_qk15_linear() raises:
    var res = qk15[func_linear](0.0, 1.0)
    assert tolerance(res.result, 0.5, 1e-10)
    print("test_qk15_linear: PASSED")


def test_qk15_quadratic() raises:
    var res = qk15[func_quadratic](0.0, 1.0)
    assert tolerance(res.result, 1.0 / 3.0, 1e-10)
    print("test_qk15_quadratic: PASSED")


def test_qk15_sin() raises:
    var res = qk15[func_sin](0.0, 1.0)
    var expected = 1.0 - cos(1.0)
    assert tolerance(res.result, expected, 1e-10)
    print("test_qk15_sin: PASSED")


def test_qng_sin() raises:
    var res = qng_integrate[func_sin](0.0, 1.0, 1e-10, 1e-10)
    var expected = 1.0 - cos(1.0)
    assert tolerance(res.val, expected, 1e-8)
    print("test_qng_sin: PASSED")


def test_qng_exp() raises:
    var res = qng_integrate[func_exp](0.0, 1.0, 1e-10, 1e-10)
    var expected = exp(1.0) - 1.0
    assert tolerance(res.val, expected, 1e-8)
    print("test_qng_exp: PASSED")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("All integration tests PASSED")
