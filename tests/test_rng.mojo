from std.testing import TestSuite
from std.math import abs

from msl.rng import RNG, MT19937


def tolerance(x: Float64, expected: Float64, tol: Float64) -> Bool:
    return abs(x - expected) < tol


def test_rng_alloc() raises:
    var rng = RNG[MT19937](MT19937(42), 42)
    assert rng.seed_val == 42
    print("test_rng_alloc: PASSED")


def test_rng_uniform() raises:
    var rng = RNG[MT19937](MT19937(42), 42)
    var val = rng.uniform()
    assert val >= 0.0
    assert val < 1.0
    print("test_rng_uniform: PASSED")


def test_rng_uniform_deterministic() raises:
    var rng1 = RNG[MT19937](MT19937(1), 1)
    var rng2 = RNG[MT19937](MT19937(1), 1)

    var v1 = rng1.uniform()
    var v2 = rng2.uniform()
    assert v1 == v2

    for _ in range(100):
        assert rng1.uniform() == rng2.uniform()

    print("test_rng_uniform_deterministic: PASSED")


def test_rng_uniform_pos() raises:
    var rng = RNG[MT19937](MT19937(42), 42)
    var val = rng.uniform_pos()
    assert val > 0.0
    assert val < 1.0

    var count = 0
    for _ in range(1000):
        if rng.uniform_pos() < 0.001:
            count += 1
    assert count > 0

    print("test_rng_uniform_pos: PASSED")


def test_rng_uniform_int() raises:
    var rng = RNG[MT19937](MT19937(42), 42)
    var val = rng.uniform_int(100)
    assert val >= 0
    assert val < 100

    var val2 = rng.uniform_int(1)
    assert val2 == 0

    print("test_rng_uniform_int: PASSED")


def test_rng_uniform_int_range() raises:
    var rng = RNG[MT19937](MT19937(789), 789)
    var counts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for _ in range(10000):
        var v = rng.uniform_int(10)
        counts[v] += 1

    for c in counts:
        assert c > 500 and c < 1500

    print("test_rng_uniform_int_range: PASSED")


def test_rng_set_seed() raises:
    var rng = RNG[MT19937](MT19937(42), 42)
    rng.set_seed(123)
    var v1 = rng.uniform()
    rng.set_seed(123)
    var v2 = rng.uniform()
    assert v1 == v2
    print("test_rng_set_seed: PASSED")


def test_rng_get() raises:
    var rng = RNG[MT19937](MT19937(999), 999)
    var val = rng.get()
    assert val != 0

    var val2 = rng.get()
    assert val != val2

    print("test_rng_get: PASSED")


def test_rng_distribution() raises:
    var rng = RNG[MT19937](MT19937(12345), 12345)
    var mean: Float64 = 0.0
    var n = 10000
    for _ in range(n):
        mean += rng.uniform()
    mean /= Float64(n)

    assert tolerance(mean, 0.5, 0.02)

    var variance: Float64 = 0.0
    for _ in range(n):
        var v = rng.uniform()
        variance += (v - mean) * (v - mean)
    variance /= Float64(n)

    assert tolerance(variance, 1.0/12.0, 0.02)

    print("test_rng_distribution: PASSED")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("All RNG tests PASSED")
