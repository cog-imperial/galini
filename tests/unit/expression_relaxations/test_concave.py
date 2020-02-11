# pylint: skip-file
import pytest
import numpy as np
import pyomo.environ as aml
from suspect.convexity import Convexity
from suspect.expression import ExpressionType
from galini.pyomo import problem_from_pyomo_model
from galini.suspect import ProblemContext
from galini.expression_relaxation.concave import UnivariateConcaveExpressionRelaxation


@pytest.fixture
def problem():
    m = aml.ConcreteModel()

    m.x = aml.Var(bounds=(2, 20))
    m.y = aml.Var(bounds=(3, 30))
    m.z = aml.Var(bounds=(4, 20))

    m.obj = aml.Objective(expr=m.x)

    m.univariate_concave = aml.Constraint(expr=aml.sqrt(m.x) >= 0)
    m.univariate_concave_2 = aml.Constraint(expr=aml.sqrt(3.0*m.x + 2.0) >= 0)
    m.univariate_concave_3 = aml.Constraint(expr=aml.sqrt(aml.log(3*m.x + 2)) >= 0)
    m.univariate_nonconcave = aml.Constraint(expr=aml.sin(m.x) >= 0)
    m.univariate_nonconcave_2 = aml.Constraint(expr=m.x**3 >= 0)
    m.univariate_nonconcave_3 = aml.Constraint(expr=3**m.x >= 0)

    m.nonunivariate = aml.Constraint(expr=m.x**m.y >= 0)
    m.nonunivariate_2 = aml.Constraint(expr=m.x + m.y + m.z >= 0)
    m.nonunivariate_3 = aml.Constraint(expr=aml.sin(m.x + aml.log(m.y)) >= 0)

    return problem_from_pyomo_model(m)


class TestUnivariateConcaveUnderestimator:
    @pytest.mark.parametrize('constraint_name',
                             ['univariate_concave', 'univariate_concave_2', 'univariate_concave_3'])
    def test_univariate_concave(self, problem, constraint_name):
        ctx = ProblemContext(problem)
        r = UnivariateConcaveExpressionRelaxation()

        constraint_expr = problem.constraint(constraint_name).root_expr

        ctx.set_convexity(constraint_expr, Convexity.Concave)

        assert r.can_relax(problem, constraint_expr, ctx)

        result = r.relax(problem, constraint_expr, ctx)

        assert result.constraints == []

    @pytest.mark.parametrize('name',
                             ['univariate_nonconcave', 'univariate_nonconcave_2',
                              'univariate_nonconcave_3'])
    def test_is_univariate(self, problem, name):
        r = UnivariateConcaveExpressionRelaxation()
        constraint_expr = problem.constraint(name).root_expr
        assert r.is_univariate(constraint_expr)

    @pytest.mark.parametrize('name',
                             ['nonunivariate', 'nonunivariate_2', 'nonunivariate_3'])
    def test_is_not_univariate(self, problem, name):
        r = UnivariateConcaveExpressionRelaxation()
        constraint_expr = problem.constraint(name).root_expr
        assert not r.is_univariate(constraint_expr)
