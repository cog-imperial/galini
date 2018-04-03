import pytest
from hypothesis import given, assume
import hypothesis.strategies as st
from galini.bound import ArbitraryPrecisionBound
from galini.math.arbitrary_precision import (
    inf, almostlte, almostgte, almosteq, pi, log, exp, cos
)
from tests.conftest import reals


@st.composite
def arbitrary_precision_bounds(draw, allow_infinity=True,
                               lower_bound=None, upper_bound=None):
    if lower_bound is None and allow_infinity:
        lower = draw(st.one_of(
            st.none(),
            reals(max_value=upper_bound, allow_infinity=False)
        ))
    else:
        lower = draw(reals(
            min_value=lower_bound,
            max_value=upper_bound,
            allow_infinity=False
        ))

    if upper_bound is None and allow_infinity:
        upper = draw(st.one_of(
            st.none(),
            reals(min_value=lower, allow_infinity=False)
        ))
    else:
        upper = draw(reals(
            min_value=lower,
            max_value=upper_bound,
            allow_infinity=False,
        ))
    return ArbitraryPrecisionBound(lower, upper)


class TestAddition(object):
    @given(arbitrary_precision_bounds())
    def test_addition_with_zero(self, bound):
        zero_bound = ArbitraryPrecisionBound(0, 0)
        assert bound + zero_bound == bound

    @given(
        arbitrary_precision_bounds(),
        arbitrary_precision_bounds(),
    )
    def test_commutative_property(self, a, b):
        assert a + b == b + a

    @given(
        arbitrary_precision_bounds(),
        arbitrary_precision_bounds(lower_bound=0),
    )
    def test_addition_with_positive_bound(self, a, b):
        c = a + b
        s = ArbitraryPrecisionBound(a.lower_bound, None)
        assert c in s

    @given(
        arbitrary_precision_bounds(),
        arbitrary_precision_bounds(upper_bound=0),
    )
    def test_addition_with_negative_bound(self, a, b):
        c = a + b
        s = ArbitraryPrecisionBound(None, a.upper_bound)
        assert c in s

    @given(
        arbitrary_precision_bounds(),
        reals(allow_infinity=False),
    )
    def test_addition_with_floats(self, a, f):
        b = ArbitraryPrecisionBound(f, f)
        assert a + f == a + b


class TestSubtraction(object):
    @given(
        arbitrary_precision_bounds(),
    )
    def test_negation(self, a):
        assert (-(-a)) == a

    @given(
        arbitrary_precision_bounds(),
    )
    def test_subtraction_with_zero(self, a):
        assert a - 0 == a

    @given(
        arbitrary_precision_bounds(),
        arbitrary_precision_bounds(lower_bound=0),
    )
    def test_subtraction_with_positive_bound(self, a, b):
        c = a - b
        s = ArbitraryPrecisionBound(None, a.upper_bound)
        assert c in s

    @given(
        arbitrary_precision_bounds(),
        arbitrary_precision_bounds(upper_bound=0),
    )
    def test_subtraction_with_negative_bound(self, a, b):
        c = a - b
        s = ArbitraryPrecisionBound(a.lower_bound, None)
        assert c in s


class TestMultiplication(object):
    @given(
        arbitrary_precision_bounds(),
    )
    def test_multiplication_with_zero(self, a):
        assert (a * 0).is_zero()

    @given(
        arbitrary_precision_bounds(),
    )
    def test_multiplication_with_one(self, a):
        assert a * 1.0 == a

    @given(
        arbitrary_precision_bounds(),
        arbitrary_precision_bounds(),
    )
    def test_commutative_property(self, a, b):
        c = a * b
        d = b * a
        assert c == d

    @given(
        arbitrary_precision_bounds(lower_bound=0),
        arbitrary_precision_bounds(lower_bound=0),
    )
    def test_multiplication_positive_with_positive(self, a, b):
        c = a * b
        assert c.is_nonnegative()

    @given(
        arbitrary_precision_bounds(lower_bound=0),
        arbitrary_precision_bounds(upper_bound=0),
    )
    def test_multiplication_positive_with_negative(self, a, b):
        c = a * b
        assert c.is_nonpositive()


class TestDivision(object):
    @given(
        arbitrary_precision_bounds(),
        arbitrary_precision_bounds(),
    )
    def test_division_by_zero(self, a, b):
        assume(b.lower_bound < 0 and b.upper_bound > 0)
        c = a / b
        assert c == ArbitraryPrecisionBound(None, None)

    @given(
        arbitrary_precision_bounds(),
    )
    def test_division_by_one(self, a):
        assert a / 1.0 == a

    @given(
        arbitrary_precision_bounds(lower_bound=0),
        arbitrary_precision_bounds(lower_bound=1),
    )
    def test_division_positive_with_positive(self, a, b):
        c = a / b
        assert c.is_nonnegative()

    @given(
        arbitrary_precision_bounds(lower_bound=0),
        arbitrary_precision_bounds(upper_bound=-1),
    )
    def test_division_positive_with_negative(self, a, b):
        c = a / b
        assert c.is_nonpositive()


class TestContains(object):
    @given(
        arbitrary_precision_bounds(),
    )
    def test_bound_contains_itself(self, a):
        assert a in a

    @given(
        arbitrary_precision_bounds(allow_infinity=False),
    )
    def test_bound_contains_midpoint(self, a):
        m = (a.lower_bound + a.upper_bound) / 2.0
        assert m in a

    @given(reals())
    def test_infinity_bound_contains_everything(self, n):
        a = ArbitraryPrecisionBound(None, None)
        assert n in a


class TestTighten(object):
    @given(
        arbitrary_precision_bounds(),
        arbitrary_precision_bounds(),
    )
    def test_bound_is_tighter(self, a, b):
        assume(a.upper_bound > b.lower_bound)
        assume(a.lower_bound < b.upper_bound)
        c = a.tighten(b)
        assert c.lower_bound >= min(a.lower_bound, b.lower_bound)
        assert c.upper_bound <= max(a.upper_bound, b.upper_bound)


class TestSize(object):
    @given(reals())
    def test_infinite_lower_bound_greater_than_everything(self, f):
        b = ArbitraryPrecisionBound(None, 0)
        assert almostgte(b.size(), f)

    @given(reals())
    def test_infinite_upper_bound_greater_than_everything(self, f):
        b = ArbitraryPrecisionBound(0, None)
        assert almostgte(b.size(), f)

    @given(reals())
    def test_infinite_bounds_greater_than_everything(self, f):
        b = ArbitraryPrecisionBound(None, None)
        assert almostgte(b.size(), f)

    @given(
        arbitrary_precision_bounds(allow_infinity=False),
        reals(allow_infinity=False, min_value=1.0)
    )
    def test_finite_bounds(self, a, f):
        b = a * f
        assert almostlte(a.size(), b.size())

    @given(
        arbitrary_precision_bounds(allow_infinity=False),
        reals(allow_infinity=False, min_value=0.0, max_value=1.0)
    )
    def test_finite_bounds_1(self, a, f):
        b = a * f
        assert almostgte(a.size(), b.size())


class TestAbs(object):
    @given(reals(max_value=0), reals(min_value=0))
    def test_bound_include_zero(self, l, u):
        b = ArbitraryPrecisionBound(l, u)

        assert almosteq(b.abs().lower_bound, 0)
        assert almosteq(
            b.abs().upper_bound,
            max(abs(l), abs(u)),
        )

    @given(reals(min_value=0), reals(min_value=0))
    def test_bound_not_include_zero(self, x, y):
        l = min(x, y)
        u = max(x, y)
        b = ArbitraryPrecisionBound(l, u)
        assert almosteq(b.abs().lower_bound, l)
        assert almosteq(b.abs().upper_bound, u)

        a = ArbitraryPrecisionBound(-u, -l)
        assert almosteq(a.abs().lower_bound, l)
        assert almosteq(a.abs().upper_bound, u)


class TestSin(object):
    def test_sin_over_2pi(self):
        b = ArbitraryPrecisionBound(-10, 10)
        assert b.sin() == ArbitraryPrecisionBound(-1, 1)

    def test_sin_contains_pi_2(self):
        b = ArbitraryPrecisionBound(0.0, 0.7*pi)
        assert b.sin() == ArbitraryPrecisionBound(0, 1)

    def test_sin_contains_3pi_2(self):
        b = ArbitraryPrecisionBound(0.7*pi, 1.7*pi)
        assert almosteq(b.sin().lower_bound, -1)

    def test_sin_contains_pi_2_and_3pi_2(self):
        b = ArbitraryPrecisionBound(0.3*pi, 1.7*pi)
        assert b.sin() == ArbitraryPrecisionBound(-1, 1)


class TestCos(object):
    def test_cos_over_2pi(self):
        b = ArbitraryPrecisionBound(-10, 10)
        assert b.cos() == ArbitraryPrecisionBound(-1, 1)

    def test_cos_contains_0(self):
        b = ArbitraryPrecisionBound(-0.2*pi, 0.7*pi)
        assert almosteq(b.cos().upper_bound, 1)

    def test_cos_contains_pi(self):
        b = ArbitraryPrecisionBound(0.7*pi, 1.2*pi)
        assert almosteq(b.cos().lower_bound, -1)

    def test_cos_contains_0_pi(self):
        b = ArbitraryPrecisionBound(-0.1*pi, 1.1*pi)
        assert b.cos() == ArbitraryPrecisionBound(-1, 1)


class TestTan(object):
    def test_tan_over_pi(self):
        b = ArbitraryPrecisionBound(-0.5*pi, 0.5*pi)
        assert b.tan() == ArbitraryPrecisionBound(None, None)

    def test_tan_contains_pi_half(self):
        b = ArbitraryPrecisionBound(0.45*pi, 0.55*pi)
        assert b.tan() == ArbitraryPrecisionBound(None, None)

    def test_tan_upper_is_pi_half(self):
        b = ArbitraryPrecisionBound(0.45*pi, 0.5*pi)
        assert b.tan().lower_bound != -inf
        assert b.tan().upper_bound == inf

    def test_tan_lower_is_pi_half(self):
        b = ArbitraryPrecisionBound(0.5*pi, 0.6*pi)
        assert b.tan().lower_bound == -inf
        assert b.tan().upper_bound != inf


class TestMonotonicFunctions(object):
    @given(reals(min_value=0.0), reals(min_value=0.0))
    def test_sqrt(self, a, b):
        b = ArbitraryPrecisionBound(min(a, b), max(a, b))
        assert almosteq(b.sqrt().lower_bound**2, b.lower_bound)
        assert almosteq(b.sqrt().upper_bound**2, b.upper_bound)

    @given(reals(min_value=0.0), reals(min_value=0.0))
    def test_exp(self, a, b):
        b = ArbitraryPrecisionBound(min(a, b), max(a, b))
        assert almosteq(log(b.exp().lower_bound), b.lower_bound)
        assert almosteq(log(b.exp().upper_bound), b.upper_bound)

    @given(reals(min_value=-1, max_value=1), reals(min_value=-1, max_value=1))
    def test_acos(self, a, b):
        b = ArbitraryPrecisionBound(min(a, b), max(a, b))
        assert almosteq(cos(b.acos().upper_bound), b.lower_bound)
