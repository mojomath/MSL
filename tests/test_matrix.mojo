from std.testing import TestSuite
from std.math import abs

from msl.matrix import Matrix, matrix_alloc, matrix_calloc, matrix_size1, matrix_size2, matrix_set_zero, matrix_set_all, matrix_set_identity, matrix_add, matrix_sub, matrix_scale, matrix_transpose, matrix_mul


def tolerance(x: Float64, expected: Float64, tol: Float64) -> Bool:
    return abs(x - expected) < tol


def test_matrix_alloc() raises:
    var mat = matrix_alloc(3, 4)
    assert matrix_size1(mat) == 3
    assert matrix_size2(mat) == 4
    print("test_matrix_alloc: PASSED")


def test_matrix_calloc() raises:
    var mat = matrix_calloc(2, 2)
    assert matrix_size1(mat) == 2
    assert matrix_size2(mat) == 2
    for i in range(2):
        for j in range(2):
            assert mat.get(i, j) == 0.0
    print("test_matrix_calloc: PASSED")


def test_matrix_get_set() raises:
    var mat = matrix_alloc(2, 2)
    mat.set(0, 0, 1.0)
    mat.set(0, 1, 2.0)
    mat.set(1, 0, 3.0)
    mat.set(1, 1, 4.0)
    assert mat.get(0, 0) == 1.0
    assert mat.get(0, 1) == 2.0
    assert mat.get(1, 0) == 3.0
    assert mat.get(1, 1) == 4.0
    print("test_matrix_get_set: PASSED")


def test_matrix_get_set_item() raises:
    var mat = matrix_alloc(2, 2)
    mat[0, 0] = 1.0
    mat[0, 1] = 2.0
    mat[1, 0] = 3.0
    mat[1, 1] = 4.0
    assert mat[0, 0] == 1.0
    assert mat[0, 1] == 2.0
    assert mat[1, 0] == 3.0
    assert mat[1, 1] == 4.0
    print("test_matrix_get_set_item: PASSED")


def test_matrix_set_zero() raises:
    var mat = matrix_alloc(3, 3)
    mat[0, 0] = 1.0
    mat[0, 1] = 2.0
    mat[1, 1] = 3.0
    matrix_set_zero(mat)
    for i in range(3):
        for j in range(3):
            assert mat.get(i, j) == 0.0
    print("test_matrix_set_zero: PASSED")


def test_matrix_set_all() raises:
    var mat = matrix_alloc(2, 3)
    matrix_set_all(mat, 7.5)
    for i in range(2):
        for j in range(3):
            assert mat.get(i, j) == 7.5
    print("test_matrix_set_all: PASSED")


def test_matrix_set_identity() raises:
    var mat = matrix_alloc(3, 3)
    matrix_set_identity(mat)
    for i in range(3):
        for j in range(3):
            if i == j:
                assert mat.get(i, j) == 1.0
            else:
                assert mat.get(i, j) == 0.0
    print("test_matrix_set_identity: PASSED")


def test_matrix_static_identity() raises:
    var mat = Matrix.identity(4)
    assert matrix_size1(mat) == 4
    assert matrix_size2(mat) == 4
    for i in range(4):
        for j in range(4):
            if i == j:
                assert mat[i, j] == 1.0
            else:
                assert mat[i, j] == 0.0
    print("test_matrix_static_identity: PASSED")


def test_matrix_static_zero() raises:
    var mat = Matrix.zero(2, 3)
    for i in range(2):
        for j in range(3):
            assert mat[i, j] == 0.0
    print("test_matrix_static_zero: PASSED")


def test_matrix_static_all() raises:
    var mat = Matrix.all(3, 2, 5.5)
    for i in range(3):
        for j in range(2):
            assert mat[i, j] == 5.5
    print("test_matrix_static_all: PASSED")


def test_matrix_ptr() raises:
    var mat = matrix_alloc(2, 2)
    mat[0, 0] = 1.0
    mat[0, 1] = 2.0
    mat[1, 0] = 3.0
    mat[1, 1] = 4.0
    
    var ptr = mat.immut_ptr()
    assert ptr[0] == 1.0
    assert ptr[1] == 2.0
    assert ptr[2] == 3.0
    assert ptr[3] == 4.0
    print("test_matrix_ptr: PASSED")


def test_matrix_large() raises:
    var n = 50
    var mat = matrix_alloc(n, n)
    matrix_set_all(mat, 42.0)
    
    for i in range(n):
        for j in range(n):
            assert mat.get(i, j) == 42.0
    print("test_matrix_large: PASSED")


def test_matrix_rectangular() raises:
    var mat = matrix_alloc(3, 5)
    var idx = 0
    for i in range(3):
        for j in range(5):
            mat.set(i, j, Float64(idx))
            idx += 1
    
    idx = 0
    for i in range(3):
        for j in range(5):
            assert mat.get(i, j) == Float64(idx)
            idx += 1
    print("test_matrix_rectangular: PASSED")


def test_matrix_nelems() raises:
    var mat = matrix_alloc(4, 5)
    assert mat.nelems() == 20
    print("test_matrix_nelems: PASSED")


def test_matrix_add() raises:
    var a = Matrix(2, 2, initialize=True)
    var b = Matrix(2, 2, initialize=True)
    a[0, 0] = 1.0; a[0, 1] = 2.0; a[1, 0] = 3.0; a[1, 1] = 4.0
    b[0, 0] = 5.0; b[0, 1] = 6.0; b[1, 0] = 7.0; b[1, 1] = 8.0
    matrix_add(a, b)
    assert a[0, 0] == 6.0 and a[0, 1] == 8.0
    assert a[1, 0] == 10.0 and a[1, 1] == 12.0
    print("test_matrix_add: PASSED")


def test_matrix_sub() raises:
    var a = Matrix(2, 2, initialize=True)
    var b = Matrix(2, 2, initialize=True)
    a[0, 0] = 5.0; a[0, 1] = 6.0; a[1, 0] = 7.0; a[1, 1] = 8.0
    b[0, 0] = 1.0; b[0, 1] = 2.0; b[1, 0] = 3.0; b[1, 1] = 4.0
    matrix_sub(a, b)
    assert a[0, 0] == 4.0 and a[0, 1] == 4.0
    assert a[1, 0] == 4.0 and a[1, 1] == 4.0
    print("test_matrix_sub: PASSED")


def test_matrix_scale() raises:
    var m = Matrix(2, 2, initialize=True)
    m[0, 0] = 1.0; m[0, 1] = 2.0; m[1, 0] = 3.0; m[1, 1] = 4.0
    matrix_scale(m, 3.0)
    assert m[0, 0] == 3.0 and m[0, 1] == 6.0
    assert m[1, 0] == 9.0 and m[1, 1] == 12.0
    print("test_matrix_scale: PASSED")


def test_matrix_transpose() raises:
    var m = Matrix(2, 3, initialize=True)
    m[0, 0] = 1.0; m[0, 1] = 2.0; m[0, 2] = 3.0
    m[1, 0] = 4.0; m[1, 1] = 5.0; m[1, 2] = 6.0
    var t = matrix_transpose(m)
    assert t.size1() == 3 and t.size2() == 2
    assert t[0, 0] == 1.0 and t[1, 0] == 2.0 and t[2, 0] == 3.0
    assert t[0, 1] == 4.0 and t[1, 1] == 5.0 and t[2, 1] == 6.0
    print("test_matrix_transpose: PASSED")


def test_matrix_mul() raises:
    var a = Matrix(2, 3, initialize=True)
    var b = Matrix(3, 2, initialize=True)
    a[0, 0] = 1.0; a[0, 1] = 2.0; a[0, 2] = 3.0
    a[1, 0] = 4.0; a[1, 1] = 5.0; a[1, 2] = 6.0
    b[0, 0] = 7.0; b[0, 1] = 8.0
    b[1, 0] = 9.0; b[1, 1] = 10.0
    b[2, 0] = 11.0; b[2, 1] = 12.0
    var c = matrix_mul(a, b)
    assert c.size1() == 2 and c.size2() == 2
    assert c[0, 0] == 58.0 and c[0, 1] == 64.0
    assert c[1, 0] == 139.0 and c[1, 1] == 154.0
    print("test_matrix_mul: PASSED")


def test_matrix_mul_identity() raises:
    var a = Matrix(3, 3, initialize=True)
    a[0, 0] = 1.0; a[0, 1] = 2.0; a[0, 2] = 3.0
    a[1, 0] = 4.0; a[1, 1] = 5.0; a[1, 2] = 6.0
    a[2, 0] = 7.0; a[2, 1] = 8.0; a[2, 2] = 9.0
    var eye = Matrix.identity(3)
    var result = matrix_mul(a, eye)
    for i in range(3):
        for j in range(3):
            assert result[i, j] == a[i, j]
    print("test_matrix_mul_identity: PASSED")


def test_matrix_borrow_ptr() raises:
    # Allocate 2x3 row-major buffer externally
    var buf = alloc[Float64](6)
    buf[0] = 1.0; buf[1] = 2.0; buf[2] = 3.0
    buf[3] = 4.0; buf[4] = 5.0; buf[5] = 6.0
    var m = Matrix(buf, 2, 3)  # non-owning view
    assert m.owner == 0
    assert m[0, 0] == 1.0 and m[0, 2] == 3.0 and m[1, 0] == 4.0 and m[1, 2] == 6.0
    # Mutations through view affect original buffer
    m[1, 1] = 99.0
    assert buf[4] == 99.0
    buf.free()
    print("test_matrix_borrow_ptr: PASSED")


def test_matrix_borrow_tda() raises:
    # Buffer with padding: 2 rows, 2 cols, but tda=4 (row stride = 4)
    var buf = alloc[Float64](8)
    for i in range(8):
        buf[i] = Float64(i)
    # Elements at [0,0]=0, [0,1]=1, [1,0]=4, [1,1]=5 (tda=4)
    var m = Matrix(buf, 2, 2, 4)  # non-owning view
    assert m[0, 0] == 0.0 and m[0, 1] == 1.0
    assert m[1, 0] == 4.0 and m[1, 1] == 5.0
    assert m.owner == 0
    buf.free()
    print("test_matrix_borrow_tda: PASSED")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("All matrix tests PASSED")
