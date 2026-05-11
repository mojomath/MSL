from std.testing import TestSuite
from std.math import abs

from msl.core import Block, block_alloc, block_calloc, block_size, block_data


def tolerance(x: Float64, expected: Float64, tol: Float64) -> Bool:
    return abs(x - expected) < tol


def test_block_alloc() raises:
    var block = block_alloc(10)
    assert block_size(block) == 10
    print("test_block_alloc: PASSED")


def test_block_calloc() raises:
    var block = block_calloc(5)
    assert block_size(block) == 5
    for i in range(5):
        assert block_data(block)[i] == 0.0
    print("test_block_calloc: PASSED")


def test_block_data() raises:
    var block = block_alloc(3)
    var data = block_data(block)
    data.store(0, 1.0)
    data.store(1, 2.0)
    data.store(2, 3.0)
    assert data[0] == 1.0
    assert data[1] == 2.0
    assert data[2] == 3.0
    print("test_block_data: PASSED")


def test_block_ptr_mut() raises:
    var block = block_alloc(3)
    var ptr = block.ptr_mut()
    ptr.store(0, 5.0)
    ptr.store(1, 10.0)
    ptr.store(2, 15.0)
    assert block_data(block)[0] == 5.0
    assert block_data(block)[1] == 10.0
    assert block_data(block)[2] == 15.0
    print("test_block_ptr_mut: PASSED")


def test_block_ptr_read() raises:
    var block = block_alloc(3)
    var ptr = block.ptr_mut()
    ptr.store(0, 1.0)
    ptr.store(1, 2.0)
    ptr.store(2, 3.0)
    
    var read_ptr = block.ptr_read()
    assert read_ptr[0] == 1.0
    assert read_ptr[1] == 2.0
    assert read_ptr[2] == 3.0
    print("test_block_ptr_read: PASSED")


def test_block_nelems() raises:
    var block = block_alloc(42)
    assert block.nelems() == 42
    print("test_block_nelems: PASSED")


def test_block_large() raises:
    var n = 1000
    var block = block_alloc(n)
    var data = block_data(block)
    for i in range(n):
        data.store(i, Float64(i))
    
    for i in range(n):
        assert data[i] == Float64(i)
    print("test_block_large: PASSED")


def test_block_fill_pattern() raises:
    var block = block_alloc(5)
    var data = block_data(block)
    for i in range(5):
        data.store(i, Float64(i) * 3.0 + 1.0)
    
    assert data[0] == 1.0
    assert data[1] == 4.0
    assert data[2] == 7.0
    assert data[3] == 10.0
    assert data[4] == 13.0
    print("test_block_fill_pattern: PASSED")


def test_block_struct() raises:
    var block = Block(10)
    assert block.size == 10
    print("test_block_struct: PASSED")


def test_block_struct_initialize() raises:
    var block = Block(5, initialize=True)
    var data = block_data(block)
    for i in range(5):
        assert data[i] == 0.0
    print("test_block_struct_initialize: PASSED")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("All block tests PASSED")
