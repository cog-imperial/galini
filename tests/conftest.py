import pytest
import hypothesis.strategies as st
from galini.pyomo import set_pyomo4_expression_tree
from galini.bound import ArbitraryPrecisionBound as Bound
from galini.context import SpecialStructurePropagationContext
import mpmath


@st.composite
def reals(draw, min_value=None, max_value=None, allow_infinity=True):
    if min_value is not None and max_value is not None:
        allow_infinity = False
    return draw(st.floats(
        min_value=min_value, max_value=max_value,
        allow_nan=False, allow_infinity=allow_infinity
    ))


@st.composite
def coefficients(draw, min_value=None, max_value=None):
    return draw(st.floats(
        min_value=min_value, max_value=max_value,
        allow_nan=False, allow_infinity=False,
    ))


class PlaceholderExpression(object):
    depth = 0


@pytest.fixture
def ctx():
    return SpecialStructurePropagationContext({})


def bound_description_to_bound(bound_str):
    if isinstance(bound_str, str):
        return {
            'zero': Bound.zero(),
            'nonpositive': Bound(None, 0),
            'nonnegative': Bound(0, None),
            'positive': Bound(1, None),
            'negative': Bound(None, -1),
            'unbounded': Bound(None, None),
        }[bound_str]
    elif isinstance(bound_str, Bound):
        return bound_str
    else:
        return Bound(bound_str, bound_str)


def pytest_sessionstart(session):
    mpmath.mp.dps = 20  # 20 decimal places precision
    set_pyomo4_expression_tree()
