# pylint: skip-file
import pytest
from galini.math.arbitrary_precision import *


class TestAlmostEq(object):
    def test_inf(self):
        assert almosteq(inf, inf)
        assert not almosteq(inf, -inf)
        assert almosteq(-inf, -inf)

    def test_finite(self):
        assert almosteq(pi, pi)
        assert not almosteq(2e20, inf)


class TestAlomstGte(object):
    def test_inf(self):
        assert almostgte(inf, -inf)
        assert almostgte(inf, inf)
        assert almostgte(-inf, -inf)

    def test_finite(self):
        assert almostgte(pi, pi)


class TestAlomstLte(object):
    def test_inf(self):
        assert almostlte(-inf, inf)
        assert almostlte(inf, inf)
        assert almostlte(-inf, -inf)

    def test_finite(self):
        assert almostlte(pi, pi)
