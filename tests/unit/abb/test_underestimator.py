# pylint: skip-file
import pytest
import numpy as np
import pyomo.environ as aml
from suspect.expression import ExpressionType
from galini.pyomo import dag_from_pyomo_model
from galini.abb.relaxation import AlphaBBRelaxation
from galini.abb.underestimator import AlphaBBExpressionRelaxation


@pytest.fixture
def problem():
    model = aml.ConcreteModel()
    model.x = aml.Var(bounds=(2, 7))
    model.y = aml.Var(bounds=(-1, 1))
    model.obj = aml.Objective(expr=0.25 * model.x * aml.sin(model.x))
    return dag_from_pyomo_model(model)


@pytest.mark.skip('Skip nonlinear.')
def test_underestimator(problem):
    abb = AlphaBBExpressionRelaxation()
    expr = problem.objective('obj').root_expr
    assert abb.can_relax(problem, expr, None)
    r = abb.relax(problem, expr, None)
    assert r.constraints == []
    new_expr = r.expression
    assert new_expr.expression_type == ExpressionType.Sum
    assert len(new_expr.children) == 3
    count_of_type = lambda expr, type_: \
        len([ch for ch in expr.children if ch.expression_type == type_])
    assert count_of_type(new_expr, ExpressionType.Product) == 1
    assert count_of_type(new_expr, ExpressionType.Quadratic) == 1
    assert count_of_type(new_expr, ExpressionType.Linear) == 1


@pytest.mark.skip('Skip nonlinear.')
def test_relaxation(problem):
    abb = AlphaBBRelaxation()
    relaxed = abb.relax(problem)

    problem_fg = problem.objectives[0].root_expr.expression_tree_data()
    relaxed_fg = relaxed.objectives[0].root_expr.expression_tree_data()

    for x in np.linspace(2, 7, 50):
        problem_fg_x = problem_fg.eval([x]).forward(0, [x])
        relaxed_fg_x = relaxed_fg.eval([x]).forward(0, [x])
        assert relaxed_fg_x <= problem_fg_x
