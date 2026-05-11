from std.testing import TestSuite

from msl.permutation import Permutation, permutation_alloc, permutation_init, permutation_next


def test_permutation_alloc() raises:
    var p = permutation_alloc(5)
    assert p.size == 5
    print("test_permutation_alloc: PASSED")


def test_permutation_init() raises:
    var p = permutation_alloc(3)
    permutation_init(p)
    assert p.get(0) == 0
    assert p.get(1) == 1
    assert p.get(2) == 2
    print("test_permutation_init: PASSED")


def test_permutation_get_set() raises:
    var p = permutation_alloc(3)
    p.set(0, 2)
    p.set(1, 0)
    p.set(2, 1)
    assert p.get(0) == 2
    assert p.get(1) == 0
    assert p.get(2) == 1
    print("test_permutation_get_set: PASSED")


def test_permutation_swap() raises:
    var p = permutation_alloc(3)
    p.set(0, 0)
    p.set(1, 1)
    p.set(2, 2)
    p.swap(0, 2)
    assert p.get(0) == 2
    assert p.get(2) == 0
    print("test_permutation_swap: PASSED")


def test_permutation_inversions() raises:
    var p = permutation_alloc(3)
    p.set(0, 2)
    p.set(1, 0)
    p.set(2, 1)
    var inv = p.inversions()
    assert inv == 3
    print("test_permutation_inversions: PASSED")


def test_permutation_inversions_sorted() raises:
    var p = permutation_alloc(4)
    permutation_init(p)
    assert p.inversions() == 0
    print("test_permutation_inversions_sorted: PASSED")


def test_permutation_inversions_reverse() raises:
    var p = permutation_alloc(4)
    p.set(0, 3)
    p.set(1, 2)
    p.set(2, 1)
    p.set(3, 0)
    assert p.inversions() == 6
    print("test_permutation_inversions_reverse: PASSED")


def test_permutation_next() raises:
    var p = permutation_alloc(3)
    permutation_init(p)
    var count = 0
    while True:
        count += 1
        if not permutation_next(p):
            break
    assert count == 6
    print("test_permutation_next: PASSED")


def test_permutation_next_sequence() raises:
    var p = permutation_alloc(3)
    permutation_init(p)
    
    var expected = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ]
    
    for k in range(6):
        for i in range(3):
            assert p.get(i) == expected[k][i]
        if k < 5:
            _ = permutation_next(p)
    print("test_permutation_next_sequence: PASSED")


def test_permutation_struct() raises:
    var p = Permutation(5)
    assert p.size == 5
    for i in range(5):
        assert p.get(i) == i
    print("test_permutation_struct: PASSED")


def test_permutation_small() raises:
    var p = permutation_alloc(1)
    permutation_init(p)
    assert p.get(0) == 0
    
    var has_next = permutation_next(p)
    assert has_next == False
    print("test_permutation_small: PASSED")


def test_permutation_swap_same() raises:
    var p = permutation_alloc(3)
    p.set(0, 1)
    p.set(1, 2)
    p.set(2, 0)
    p.swap(1, 1)
    assert p.get(1) == 2
    print("test_permutation_swap_same: PASSED")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("All permutation tests PASSED")
