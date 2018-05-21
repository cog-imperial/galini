import pytest
import hypothesis.strategies as st


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
