# pylint: skip-file
import pytest
import pyomo.environ as aml
from galini.pyomo import problem_from_pyomo_model
from galini.suspect import ProblemContext
from galini.expression_relaxation.linear import LinearExpressionRelaxation


@pytest.fixture
def problem():
    m = aml.ConcreteModel()

    m.I = range(10)
    m.x = aml.Var(m.I, bounds=(-2, 2))
    m.y = aml.Var(bounds=(-3, 3))

    m.obj = aml.Objective(expr=m.y)

    m.linear = aml.Constraint(expr=sum(m.x[i] for i in m.I) >= 0)
    m.not_linear = aml.Constraint(expr=m.y * sum(m.x[i] for i in m.I) >= 0)

    return problem_from_pyomo_model(m)


class TestLinearUnderestimator:
    def test_linear_terms(self, problem):
        ctx = ProblemContext(problem)
        r = LinearExpressionRelaxation()

        linear = problem.constraint('linear').root_expr

        assert r.can_relax(problem, linear, ctx)
        result = r.relax(problem, linear, ctx)

        assert result.constraints == []
        assert result.expression == linear

    def test_nonlinear_terms(self, problem):
        ctx = ProblemContext(problem)
        r = LinearExpressionRelaxation()

        nonlinear = problem.constraint('not_linear').root_expr

        assert not r.can_relax(problem, nonlinear, ctx)
