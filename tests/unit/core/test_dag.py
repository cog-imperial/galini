# pylint: skip-file
import pytest
import numpy as np
from galini.core import *
from suspect.expression import ExpressionType, UnaryFunctionType


@pytest.fixture
def problem():
    return RootProblem('test')


def test_problem_creation(problem):
    assert problem.size == 0
    assert problem.max_depth() == 0


def test_insert_variable(problem):
    problem.add_variable('x', None, None, Domain.REAL)
    assert problem.max_depth() == 0
    with pytest.raises(Exception):
        problem.add_variable('x', v, None, None, Domain.INTEGER)


def test_insert_constraint(problem):
    v = problem.add_variable('x', None, None, Domain.REAL)
    problem.add_constraint('c0', v, None, None)
    problem.add_constraint('c1', v, 0.0, 1.0)
    assert problem.num_constraints == 2
    with pytest.raises(Exception):
        problem.add_constraint('c0', v, None, None)


def test_insert_objective(problem):
    v = problem.add_variable('x', None, None, Domain.REAL)
    problem.add_objective('o0', v, Sense.MINIMIZE)
    problem.add_objective('o1', v, Sense.MAXIMIZE)
    assert problem.num_objectives == 2
    with pytest.raises(Exception):
        problem.add_objective('o0', v, Sense.MINIMIZE)


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
    v0 = problem.add_variable('x', None, None, Domain.REAL)
    problem.insert_vertex(c0)
    s0 = SumExpression([c0, v0])
    problem.insert_vertex(s0)
    assert s0.idx == 2
    assert s0.nth_children(0) == c0
    assert s0.nth_children(1) == v0
    problem.add_variable('y', None, None, Domain.REAL)
    assert s0.idx == 3
    assert s0.nth_children(0) == c0
    assert s0.nth_children(1) == v0
    assert problem.vertex_depth(s0.idx) == 2
    assert problem.vertex_depth(c0.idx) == 1
    assert problem.size == 4


@pytest.mark.parametrize('expr,expected', [
    (ProductExpression([Variable(), Variable()]), ExpressionType.Product),
    (DivisionExpression([Variable(), Variable()]), ExpressionType.Division),
    (SumExpression([Variable(), Variable(), Variable()]), ExpressionType.Sum),
    (PowExpression([Variable(), Variable()]), ExpressionType.Power),
    (LinearExpression([Variable()], np.array([1.0]), 0.0), ExpressionType.Linear),
    (NegationExpression([Variable()]), ExpressionType.Negation),
    (Variable(), ExpressionType.Variable),
    (Constant(0.0), ExpressionType.Constant),
])
def test_suspect_expression_types(expr, expected):
    assert expr.expression_type == expected

@pytest.mark.parametrize('expr,expected', [
    (AbsExpression([Variable()]), UnaryFunctionType.Abs),
    (SqrtExpression([Variable()]), UnaryFunctionType.Sqrt),
    (ExpExpression([Variable()]), UnaryFunctionType.Exp),
    (LogExpression([Variable()]), UnaryFunctionType.Log),
    (SinExpression([Variable()]), UnaryFunctionType.Sin),
    (CosExpression([Variable()]), UnaryFunctionType.Cos),
    (TanExpression([Variable()]), UnaryFunctionType.Tan),
    (AsinExpression([Variable()]), UnaryFunctionType.Asin),
    (AcosExpression([Variable()]), UnaryFunctionType.Acos),
    (AtanExpression([Variable()]), UnaryFunctionType.Atan),
])
def test_suspect_unary_function_type(expr, expected):
    assert expr.func_type == expected
