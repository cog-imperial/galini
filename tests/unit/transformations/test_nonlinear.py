# pylint: skip-file
import pytest
import numpy as np
import pyomo.environ as aml
from suspect.interval import Interval
from suspect.expression import ExpressionType, UnaryFunctionType
from galini.pyomo import dag_from_pyomo_model
from galini.special_structure import detect_special_structure
from galini.transformations.nonlinear import ReplaceNonlinearTransformation
from galini.suspect import ProblemContext


@pytest.fixture
def problem():
    m = aml.ConcreteModel()

    m.x = aml.Var(bounds=(-2, 2))
    m.y = aml.Var(bounds=(-3, 3))
    m.z = aml.Var(bounds=(-4, 2))

    m.obj = aml.Objective(expr=m.x)

    m.nonlinear_1_child = aml.Constraint(expr=aml.exp(m.x) >= 0)
    m.nonlinear_multiple_children = aml.Constraint(expr=aml.exp(m.x) + aml.log(m.y) >= 0)
    m.nonlinear_linear = aml.Constraint(expr=aml.exp(m.x) + (2.0*m.y + 3.0*m.z + 4.0*m.x + 2.0) >= 0)

    return dag_from_pyomo_model(m)


def test_nonlinear_with_1_child(problem):
    t = ReplaceNonlinearTransformation(problem, None)
    ctx = ProblemContext(problem)
    root_expr = problem.constraint('nonlinear_1_child').root_expr
    result = t.apply(root_expr, ctx)
    assert result.expression.expression_type == ExpressionType.UnaryFunction
    assert result.expression.uid == root_expr.uid
    assert result.constraints == []


def test_nonlinear_with_multiple_children(problem):
    t = ReplaceNonlinearTransformation(problem, None)
    ctx = detect_special_structure(problem)
    root_expr = problem.constraint('nonlinear_multiple_children').root_expr
    ctx.set_bounds(root_expr, Interval(0, 10))

    result = t.apply(root_expr, ctx)

    assert result.expression.expression_type == ExpressionType.Linear
    linear_expr = result.expression
    assert len(result.constraints) == 2
    c0, c1 = result.constraints

    if _is_func_type(c0.root_expr, UnaryFunctionType.Exp):
        assert _is_func_type(c0.root_expr, UnaryFunctionType.Exp)
        assert _is_func_type(c1.root_expr, UnaryFunctionType.Log)
    else:
        assert _is_func_type(c0.root_expr, UnaryFunctionType.Log)
        assert _is_func_type(c1.root_expr, UnaryFunctionType.Exp)


def test_nonlienar_with_linear(problem):
    t = ReplaceNonlinearTransformation(problem, None)
    ctx = detect_special_structure(problem)
    root_expr = problem.constraint('nonlinear_linear').root_expr
    ctx.set_bounds(root_expr, Interval(0, 10))

    result = t.apply(root_expr, ctx)

    assert len(result.constraints) == 1
    assert _is_func_type(result.constraints[0].root_expr, UnaryFunctionType.Exp)
    expr = result.expression

    assert len(expr.children) == 4
    assert np.isclose(expr.constant_term, 2.0)

    coefficients = np.sort([expr.coefficient(ch) for ch in expr.children])
    assert np.allclose(coefficients, [1.0, 2.0, 3.0, 4.0])


def _is_func_type(expr, type_):
    assert len(expr.children) == 2
    first, second = expr.children
    if first.expression_type == ExpressionType.UnaryFunction:
        return first.func_type == type_
    assert second.expression_type == ExpressionType.UnaryFunction
    return second.func_type == type_
