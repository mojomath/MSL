from std.testing import TestSuite
from std.math import abs, cos, sin, exp

from msl.ode import ode_rk4, ode_rkf45

comptime MutExt = MutExternalOrigin


def tolerance(x: Float64, expected: Float64, tol: Float64) -> Bool:
    return abs(x - expected) < tol


# ===----------------------------------------------------------------------=== #
# Test systems
# ===----------------------------------------------------------------------=== #
#
# Harmonic oscillator: dy0/dt = y1,  dy1/dt = -y0
# Exact: y0(t) = cos(t),  y1(t) = -sin(t)  with y0(0)=1, y1(0)=0
#
# Exponential decay: dy/dt = -y
# Exact: y(t) = exp(-t)  with y(0) = 1


# ===----------------------------------------------------------------------=== #
# ode_rk4 tests
# ===----------------------------------------------------------------------=== #


def test_rk4_exponential() raises:
    def rhs[origin_fn_y: MutOrigin, origin_fn_dydt: MutOrigin, //](
        t: Float64,
        y: UnsafePointer[Float64, origin_fn_y],
        dydt: UnsafePointer[Float64, origin_fn_dydt],
    ) capturing:
        dydt[0] = -y[0]

    var y = alloc[Float64](1)
    y[0] = 1.0
    var r = ode_rk4[rhs](0.0, 1.0, 0.01, y, 1)
    assert r.success
    assert tolerance(y[0], exp(-1.0), 1e-7)
    y.free()
    print("test_rk4_exponential: PASSED")


def test_rk4_harmonic() raises:
    def rhs[origin_fn_y: MutOrigin, origin_fn_dydt: MutOrigin, //](
        t: Float64,
        y: UnsafePointer[Float64, origin_fn_y],
        dydt: UnsafePointer[Float64, origin_fn_dydt],
    ) capturing:
        dydt[0] = y[1]
        dydt[1] = -y[0]

    var y = alloc[Float64](2)
    y[0] = 1.0; y[1] = 0.0
    var r = ode_rk4[rhs](0.0, 1.0, 0.01, y, 2)
    assert r.success
    assert tolerance(y[0], cos(1.0), 1e-7)
    assert tolerance(y[1], -sin(1.0), 1e-7)
    y.free()
    print("test_rk4_harmonic: PASSED")


def test_rk4_longer_interval() raises:
    def rhs[origin_fn_y: MutOrigin, origin_fn_dydt: MutOrigin, //](
        t: Float64,
        y: UnsafePointer[Float64, origin_fn_y],
        dydt: UnsafePointer[Float64, origin_fn_dydt],
    ) capturing:
        dydt[0] = y[1]
        dydt[1] = -y[0]

    var y = alloc[Float64](2)
    y[0] = 1.0; y[1] = 0.0
    var r = ode_rk4[rhs](0.0, 3.141592653589793, 0.001, y, 2)
    assert r.success
    # y0(pi) = cos(pi) = -1
    assert tolerance(y[0], -1.0, 1e-6)
    y.free()
    print("test_rk4_longer_interval: PASSED")


def test_rk4_nsteps() raises:
    def rhs[origin_fn_y: MutOrigin, origin_fn_dydt: MutOrigin, //](
        t: Float64,
        y: UnsafePointer[Float64, origin_fn_y],
        dydt: UnsafePointer[Float64, origin_fn_dydt],
    ) capturing:
        dydt[0] = -y[0]

    var y = alloc[Float64](1)
    y[0] = 1.0
    var r = ode_rk4[rhs](0.0, 1.0, 0.1, y, 1)
    assert r.success and r.nsteps == 10 and r.nfev == 40
    y.free()
    print("test_rk4_nsteps: PASSED")


def test_rk4_captures_closure() raises:
    var omega: Float64 = 2.0

    def rhs[origin_fn_y: MutOrigin, origin_fn_dydt: MutOrigin, //](
        t: Float64,
        y: UnsafePointer[Float64, origin_fn_y],
        dydt: UnsafePointer[Float64, origin_fn_dydt],
    ) capturing:
        dydt[0] = y[1]
        dydt[1] = -(omega * omega) * y[0]

    # y'' + 4y = 0, y(0)=1, y'(0)=0 → y(t) = cos(2t)
    var y = alloc[Float64](2)
    y[0] = 1.0; y[1] = 0.0
    var r = ode_rk4[rhs](0.0, 1.0, 0.01, y, 2)
    assert r.success
    assert tolerance(y[0], cos(2.0), 1e-6)
    y.free()
    print("test_rk4_captures_closure: PASSED")


# ===----------------------------------------------------------------------=== #
# ode_rkf45 tests
# ===----------------------------------------------------------------------=== #


def test_rkf45_exponential() raises:
    def rhs[origin_fn_y: MutOrigin, origin_fn_dydt: MutOrigin, //](
        t: Float64,
        y: UnsafePointer[Float64, origin_fn_y],
        dydt: UnsafePointer[Float64, origin_fn_dydt],
    ) capturing:
        dydt[0] = -y[0]

    var y = alloc[Float64](1)
    y[0] = 1.0
    var r = ode_rkf45[rhs](0.0, 1.0, 0.1, y, 1, epsabs=1e-8, epsrel=1e-8)
    assert r.success
    assert tolerance(y[0], exp(-1.0), 1e-7)
    y.free()
    print("test_rkf45_exponential: PASSED")


def test_rkf45_harmonic() raises:
    def rhs[origin_fn_y: MutOrigin, origin_fn_dydt: MutOrigin, //](
        t: Float64,
        y: UnsafePointer[Float64, origin_fn_y],
        dydt: UnsafePointer[Float64, origin_fn_dydt],
    ) capturing:
        dydt[0] = y[1]
        dydt[1] = -y[0]

    var y = alloc[Float64](2)
    y[0] = 1.0; y[1] = 0.0
    var r = ode_rkf45[rhs](0.0, 1.0, 0.1, y, 2, epsabs=1e-8, epsrel=1e-8)
    assert r.success
    assert tolerance(y[0], cos(1.0), 1e-7)
    assert tolerance(y[1], -sin(1.0), 1e-7)
    y.free()
    print("test_rkf45_harmonic: PASSED")


def test_rkf45_fewer_steps_than_rk4() raises:
    # For the same problem, adaptive should use fewer steps than fixed h=0.01
    def rhs[origin_fn_y: MutOrigin, origin_fn_dydt: MutOrigin, //](
        t: Float64,
        y: UnsafePointer[Float64, origin_fn_y],
        dydt: UnsafePointer[Float64, origin_fn_dydt],
    ) capturing:
        dydt[0] = -y[0]

    var y1 = alloc[Float64](1); y1[0] = 1.0
    var r_rk4 = ode_rk4[rhs](0.0, 1.0, 0.01, y1, 1)

    var y2 = alloc[Float64](1); y2[0] = 1.0
    var r_rkf45 = ode_rkf45[rhs](0.0, 1.0, 0.1, y2, 1, epsabs=1e-6, epsrel=1e-6)

    assert r_rkf45.success
    assert r_rkf45.nsteps < r_rk4.nsteps
    y1.free(); y2.free()
    print("test_rkf45_fewer_steps_than_rk4: PASSED")


def test_rkf45_captures_closure() raises:
    var k: Float64 = 3.0

    def rhs[origin_fn_y: MutOrigin, origin_fn_dydt: MutOrigin, //](
        t: Float64,
        y: UnsafePointer[Float64, origin_fn_y],
        dydt: UnsafePointer[Float64, origin_fn_dydt],
    ) capturing:
        dydt[0] = -k * y[0]

    # y(t) = exp(-3t)
    var y = alloc[Float64](1)
    y[0] = 1.0
    var r = ode_rkf45[rhs](0.0, 1.0, 0.1, y, 1, epsabs=1e-8, epsrel=1e-8)
    assert r.success
    assert tolerance(y[0], exp(-3.0), 1e-7)
    y.free()
    print("test_rkf45_captures_closure: PASSED")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("All ODE tests PASSED")
