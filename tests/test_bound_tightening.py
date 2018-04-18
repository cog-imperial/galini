import pytest
from unittest.mock import MagicMock
from hypothesis import given, assume
import hypothesis.strategies as st
from galini.bound.tightening import *
import galini.dag.expressions as dex
from galini.bound import ArbitraryPrecisionBound as Bound
from tests.conftest import (
    PlaceholderExpression,
    bound_description_to_bound,
    coefficients,
    reals,
    ctx,
)


@pytest.fixture
def visitor():
    return BoundsTighteningVisitor()


@given(reals(), reals(), reals(), reals())
def test_constraint_bound(visitor, ctx, a, b, c, d):
    e_lb, e_ub = min(a, b), max(a, b)
    c_lb, c_ub = min(c, d), max(c, d)
    assume(max(e_lb, c_lb) <= min(e_ub, c_ub))
    p = PlaceholderExpression()
    ctx.bound[p] = Bound(e_lb, e_ub)
    cons = dex.Constraint('c0', lower_bound=c_lb, upper_bound=c_ub, children=[p])
    visitor(cons, ctx)
    expected = Bound(max(e_lb, c_lb), min(c_ub, e_ub))
    assert ctx.bound[p] == expected


def test_sum_bound(visitor, ctx):
    # build `lb <= a + b + c <= ub`
    a = PlaceholderExpression()
    b = PlaceholderExpression()
    c = PlaceholderExpression()

    ctx.bound[a] = Bound(0, 10)
    ctx.bound[b] = Bound(0, 200)
    ctx.bound[c] = Bound(0, 10)

    sum_ = dex.SumExpression(children=[a, b, c])
    ctx.bound[sum_] = Bound(0, 100)

    visitor(sum_, ctx)
    assert ctx.bound[b] == Bound(0, 100)


def test_linear_bound(visitor, ctx):
    # build `lb <= a + b + c <= ub`
    a = PlaceholderExpression()
    b = PlaceholderExpression()
    c = PlaceholderExpression()

    ctx.bound[a] = Bound(0, 10)
    ctx.bound[b] = Bound(0, 200)
    ctx.bound[c] = Bound(0, 10)

    sum_ = dex.LinearExpression(
        children=[a, b, c],
        coefficients=[2.0, -3.0, -1.0],
        constant_term=200.0,
    )
    ctx.bound[sum_] = Bound(0, 100)

    visitor(sum_, ctx)
    assert ctx.bound[b].lower_bound == 30.0
    assert ctx.bound[b].upper_bound > 73.0
