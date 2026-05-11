from std.testing import TestSuite
from std.math import abs

from msl.vector import Vector, vector_alloc, vector_calloc, vector_size, vector_stride, vector_set_zero, vector_set_all, vector_add, vector_sub, vector_scale, vector_axpy, vector_dot, vector_norm


def tolerance(x: Float64, expected: Float64, tol: Float64) -> Bool:
    return abs(x - expected) < tol


def test_vector_alloc() raises:
    var vec = vector_alloc(10)
    assert vector_size(vec) == 10
    print("test_vector_alloc: PASSED")


def test_vector_calloc() raises:
    var vec = vector_calloc(5)
    assert vector_size(vec) == 5
    for i in range(5):
        assert vec.get(i) == 0.0
    print("test_vector_calloc: PASSED")


def test_vector_get_set() raises:
    var vec = vector_alloc(3)
    vec.set(0, 1.5)
    vec.set(1, 2.5)
    vec.set(2, 3.5)
    assert vec.get(0) == 1.5
    assert vec.get(1) == 2.5
    assert vec.get(2) == 3.5
    print("test_vector_get_set: PASSED")


def test_vector_get_set_item() raises:
    var vec = vector_alloc(3)
    vec[0] = 1.5
    vec[1] = 2.5
    vec[2] = 3.5
    assert vec[0] == 1.5
    assert vec[1] == 2.5
    assert vec[2] == 3.5
    print("test_vector_get_set_item: PASSED")


def test_vector_set_zero() raises:
    var vec = vector_alloc(3)
    vec.set(0, 1.0)
    vec.set(1, 2.0)
    vec.set(2, 3.0)
    vector_set_zero(vec)
    assert vec.get(0) == 0.0
    assert vec.get(1) == 0.0
    assert vec.get(2) == 0.0
    print("test_vector_set_zero: PASSED")


def test_vector_set_all() raises:
    var vec = vector_alloc(3)
    vector_set_all(vec, 5.0)
    assert vec.get(0) == 5.0
    assert vec.get(1) == 5.0
    assert vec.get(2) == 5.0
    print("test_vector_set_all: PASSED")


def test_vector_stride() raises:
    var vec = vector_alloc(5)
    assert vector_stride(vec) == 1
    print("test_vector_stride: PASSED")


def test_vector_stride_custom() raises:
    var vec = Vector(5, stride=2)
    assert vector_stride(vec) == 2
    vec.set(0, 1.0)
    vec.set(1, 2.0)
    assert vec.get(0) == 1.0
    assert vec.get(2) == 2.0
    print("test_vector_stride_custom: PASSED")


def test_vector_ptr() raises:
    var vec = vector_alloc(3)
    vec.set(0, 1.0)
    vec.set(1, 2.0)
    vec.set(2, 3.0)

    var ptr = vec.ptr_read()
    assert ptr[0] == 1.0
    assert ptr[1] == 2.0
    assert ptr[2] == 3.0
    print("test_vector_ptr: PASSED")


def test_vector_large() raises:
    var n = 1000
    var vec = vector_alloc(n)
    vector_set_all(vec, 42.0)

    for i in range(n):
        assert vec.get(i) == 42.0
    print("test_vector_large: PASSED")


def test_vector_filled_pattern() raises:
    var vec = vector_alloc(5)
    for i in range(5):
        vec.set(i, Float64(i) * 2.0)

    assert vec.get(0) == 0.0
    assert vec.get(1) == 2.0
    assert vec.get(2) == 4.0
    assert vec.get(3) == 6.0
    assert vec.get(4) == 8.0
    print("test_vector_filled_pattern: PASSED")


def test_vector_add() raises:
    var a = Vector(3, initialize=True)
    var b = Vector(3, initialize=True)
    a[0] = 1.0; a[1] = 2.0; a[2] = 3.0
    b[0] = 4.0; b[1] = 5.0; b[2] = 6.0
    vector_add(a, b)
    assert a[0] == 5.0 and a[1] == 7.0 and a[2] == 9.0
    print("test_vector_add: PASSED")


def test_vector_sub() raises:
    var a = Vector(3, initialize=True)
    var b = Vector(3, initialize=True)
    a[0] = 5.0; a[1] = 7.0; a[2] = 9.0
    b[0] = 1.0; b[1] = 2.0; b[2] = 3.0
    vector_sub(a, b)
    assert a[0] == 4.0 and a[1] == 5.0 and a[2] == 6.0
    print("test_vector_sub: PASSED")


def test_vector_scale() raises:
    var v = Vector(3, initialize=True)
    v[0] = 1.0; v[1] = 2.0; v[2] = 3.0
    vector_scale(v, 2.0)
    assert v[0] == 2.0 and v[1] == 4.0 and v[2] == 6.0
    print("test_vector_scale: PASSED")


def test_vector_axpy() raises:
    var x = Vector(3, initialize=True)
    var y = Vector(3, initialize=True)
    x[0] = 1.0; x[1] = 2.0; x[2] = 3.0
    y[0] = 10.0; y[1] = 10.0; y[2] = 10.0
    vector_axpy(2.0, x, y)
    assert y[0] == 12.0 and y[1] == 14.0 and y[2] == 16.0
    print("test_vector_axpy: PASSED")


def test_vector_dot() raises:
    var a = Vector(3, initialize=True)
    var b = Vector(3, initialize=True)
    a[0] = 1.0; a[1] = 2.0; a[2] = 3.0
    b[0] = 4.0; b[1] = 5.0; b[2] = 6.0
    var d = vector_dot(a, b)
    assert d == 32.0
    print("test_vector_dot: PASSED")


def test_vector_norm() raises:
    var v = Vector(3, initialize=True)
    v[0] = 3.0; v[1] = 4.0; v[2] = 0.0
    var n = vector_norm(v)
    assert n == 5.0
    print("test_vector_norm: PASSED")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("All vector tests PASSED")
