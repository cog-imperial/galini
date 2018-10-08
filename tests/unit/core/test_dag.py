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


VARIABLE = Variable('x', None, None, None)

@pytest.mark.parametrize('expr,expected', [
    (ProductExpression([VARIABLE, VARIABLE]), ExpressionType.Product),
    (DivisionExpression([VARIABLE, VARIABLE]), ExpressionType.Division),
    (SumExpression([VARIABLE, VARIABLE, VARIABLE]), ExpressionType.Sum),
    (PowExpression([VARIABLE, VARIABLE]), ExpressionType.Power),
    (LinearExpression([VARIABLE], np.array([1.0]), 0.0), ExpressionType.Linear),
    (NegationExpression([VARIABLE]), ExpressionType.Negation),
    (VARIABLE, ExpressionType.Variable),
    (Constant(0.0), ExpressionType.Constant),
])
def test_suspect_expression_types(expr, expected):
    assert expr.expression_type == expected

@pytest.mark.parametrize('expr,expected', [
    (AbsExpression([VARIABLE]), UnaryFunctionType.Abs),
    (SqrtExpression([VARIABLE]), UnaryFunctionType.Sqrt),
    (ExpExpression([VARIABLE]), UnaryFunctionType.Exp),
    (LogExpression([VARIABLE]), UnaryFunctionType.Log),
    (SinExpression([VARIABLE]), UnaryFunctionType.Sin),
    (CosExpression([VARIABLE]), UnaryFunctionType.Cos),
    (TanExpression([VARIABLE]), UnaryFunctionType.Tan),
    (AsinExpression([VARIABLE]), UnaryFunctionType.Asin),
    (AcosExpression([VARIABLE]), UnaryFunctionType.Acos),
    (AtanExpression([VARIABLE]), UnaryFunctionType.Atan),
])
def test_suspect_unary_function_type(expr, expected):
    assert expr.func_type == expected
