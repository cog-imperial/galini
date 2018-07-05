# pylint: skip-file
import pytest
import numpy as np
from galini.core import *
from suspect.expression import ExpressionType, UnaryFunctionType


@pytest.fixture
def problem():
    return Problem('test')

def test_problem_creation(problem):
    assert problem.size == 0
    assert problem.max_depth() == 0


def test_insert_variable(problem):
    problem.add_variable('x', None, None, Domain.REALS)
    assert problem.max_depth() == 0


def test_constants_minimum_depth(problem):
    problem.insert_vertex(Constant(1.234))
    assert problem.max_depth() == 1


def test_vertex_index_is_updated(problem):
    cs = [Constant(i) for i in range(10)]
    for c in cs:
        problem.insert_vertex(c)
    assert problem.max_depth() == 1
    assert problem.size == 10
    for i, c in enumerate(cs):
        assert c.idx == i


def test_children_are_updated(problem):
    c0 = Constant(1.234)
    v0 = problem.add_variable('x', None, None, Domain.REALS)
    problem.insert_vertex(c0)
    s0 = SumExpression([c0.idx, v0.idx])
    problem.insert_vertex(s0)
    assert s0.idx == 2
    assert s0.nth_children(0) == 1
    assert s0.nth_children(1) == 0
    problem.add_variable('y', None, None, Domain.REALS)
    assert s0.idx == 3
    assert s0.nth_children(0) == 2
    assert s0.nth_children(1) == 0
    assert problem.vertex_depth(s0.idx) == 2
    assert problem.vertex_depth(c0.idx) == 1
    assert problem.size == 4


@pytest.mark.parametrize('expr,expected', [
    (ProductExpression([0, 1]), ExpressionType.Product),
    (DivisionExpression([0, 1]), ExpressionType.Division),
    (SumExpression([0, 1]), ExpressionType.Sum),
    (PowExpression([0, 1]), ExpressionType.Power),
    (LinearExpression([0], np.array([1.0]), 0.0), ExpressionType.Linear),
    (NegationExpression([0]), ExpressionType.Negation),
    (Variable(None, None, Domain.REALS), ExpressionType.Variable),
    (Constant(0.0), ExpressionType.Constant),
])
def test_suspect_expression_types(expr, expected):
    assert expr.expression_type == expected

@pytest.mark.parametrize('expr,expected', [
    (AbsExpression([0]), UnaryFunctionType.Abs),
    (SqrtExpression([0]), UnaryFunctionType.Sqrt),
    (ExpExpression([0]), UnaryFunctionType.Exp),
    (LogExpression([0]), UnaryFunctionType.Log),
    (SinExpression([0]), UnaryFunctionType.Sin),
    (CosExpression([0]), UnaryFunctionType.Cos),
    (TanExpression([0]), UnaryFunctionType.Tan),
    (AsinExpression([0]), UnaryFunctionType.Asin),
    (AcosExpression([0]), UnaryFunctionType.Acos),
    (AtanExpression([0]), UnaryFunctionType.Atan),
])
def test_suspect_unary_function_type(expr, expected):
    assert expr.func_type == expected
