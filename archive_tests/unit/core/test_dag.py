# pylint: skip-file
import pytest
import numpy as np
from galini.core import Domain
from galini.core._core import *
from suspect.expression import ExpressionType, UnaryFunctionType


@pytest.fixture
def graph():
    return Graph()


def test_problem_creation(graph):
    assert len(graph) == 0
    assert list(graph) == []


def test_insert_variable(graph):
    graph.insert_vertex(Variable('x', None, None, Domain.REAL))
    assert graph.max_depth() == 0


def test_constants_minimum_depth(graph):
    graph.insert_vertex(Constant(1.234))
    assert graph.max_depth() == 2


def test_vertex_index_is_updated(graph):
    cs = [Constant(i) for i in range(10)]
    for c in cs:
        graph.insert_vertex(c)
    assert graph.max_depth() == 2
    assert len(graph) == 10
    for i, c in enumerate(cs):
        assert c.idx == i


def test_children_are_updated(graph):
    c0 = Constant(1.234)
    v0 = Variable('x', None, None, Domain.REAL)
    graph.insert_vertex(c0)
    graph.insert_vertex(v0)
    s0 = SumExpression([c0, v0])
    graph.insert_vertex(s0)
    assert s0.idx == 2
    assert s0.nth_children(0) == c0
    assert s0.nth_children(1) == v0
    v1 = Variable('y', None, None, Domain.REAL)
    graph.insert_vertex(v1)
    assert s0.idx == 3
    assert s0.nth_children(0) == c0
    assert s0.nth_children(1) == v0
    vertices = list(graph)
    assert vertices[s0.idx].depth == 3
    assert vertices[c0.idx].depth == 2
    assert len(graph) == 4


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
