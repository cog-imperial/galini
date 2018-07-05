import pytest
from suspect.math import make_number
from galini.pyomo.float_hash import BTreeFloatHasher, RoundFloatHasher


def test_btree_float_hasher():
    hasher = BTreeFloatHasher()

    # fill hasher with some numbers
    for i in range(100):
        hasher.hash(make_number(i) / 10.0)
        hasher.hash(make_number(-i) / 5.0)

    h1 = hasher.hash(10.123)
    h2 = hasher.hash(10.123)
    h3 = hasher.hash(10.1234)
    h4 = hasher.hash(10.123000000001)

    assert h1 == h2
    assert h2 != h3
    assert h2 != h4


def test_round_float_hasher():
    hasher = RoundFloatHasher(3)
    assert hasher.hash(100.123) == hasher.hash(100.123)
    assert hasher.hash(100.123) == hasher.hash(100.123456)
    assert hasher.hash(10) == hasher.hash(10.0001)
