# pylint: skip-file
import pytest
import pyomo.environ as aml
from suspect.expression import ExpressionType
from galini.pyomo import dag_from_pyomo_model
from galini.abb.underestimator import AlphaBBUnderestimator


@pytest.fixture
def problem():
    model = aml.ConcreteModel()
    model.x = aml.Var(bounds=(2, 7))
    model.y = aml.Var(bounds=(-1, 1))
    model.obj = aml.Objective(expr=0.25 * model.x * aml.sin(model.x))
    return dag_from_pyomo_model(model)


def test_underestimator(problem):
    abb = AlphaBBUnderestimator()
    expr = problem.objective('obj').root_expr
    assert abb.can_underestimate(problem, expr, None)
    r = abb.underestimate(problem, expr, None)
    assert r.constraints == []
    new_expr = r.expression
    assert new_expr.expression_type == ExpressionType.Sum
    assert len(new_expr.children) == 3
    count_of_type = lambda expr, type_: \
        len([ch for ch in expr.children if ch.expression_type == type_])
    print(new_expr.children)
    assert count_of_type(new_expr, ExpressionType.Product) == 2
    assert count_of_type(new_expr, ExpressionType.Linear) == 1
