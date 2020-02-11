# pylint: skip-file
import pytest
import numpy as np
import pyomo.environ as aml
from suspect.polynomial import PolynomialDegree
from suspect.expression import ExpressionType
from galini.pyomo import problem_from_pyomo_model
from galini.suspect import ProblemContext
from galini.expression_relaxation.bilinear import McCormickExpressionRelaxation
from galini.suspect import ProblemContext


@pytest.fixture
def problem():
    m = aml.ConcreteModel()

    m.x = aml.Var(bounds=(-2, 2))
    m.y = aml.Var(bounds=(-3, 3))
    m.z = aml.Var(bounds=(-4, 2))

    m.obj = aml.Objective(expr=m.x)

    m.bilinear = aml.Constraint(expr=m.x*m.y >= 0)
    m.bilinear_coef = aml.Constraint(expr=5.0*m.x*m.y >= 0)
    m.bilinear_coef_2 = aml.Constraint(expr=m.x*m.y*6.0 >= 0)
    m.bilinear_sum = aml.Constraint(expr=2*m.x*m.y + 3*m.x*m.z + 4*m.y*m.z >= 0)
    m.trilinear = aml.Constraint(expr=m.x*m.y*m.z >= 0)
    m.power = aml.Constraint(expr=m.x**2 >= 0)

    return problem_from_pyomo_model(m)


class TestMcCormickUnderestimator:
    def test_bilinear_terms(self, problem):
        r = McCormickExpressionRelaxation()
        ctx = ProblemContext(problem)
        bilinear = problem.constraint('bilinear').root_expr

        assert r.can_relax(problem, bilinear, ctx)

        result = r.relax(problem, bilinear, ctx)
        self._check_constraints(result)
        assert result.expression.expression_type == ExpressionType.Linear
        assert len(result.expression.children) == 1
        aux_var = result.expression.children[0]
        assert aux_var.is_auxiliary

    def test_bilinear_with_coef(self, problem):
        ctx = ProblemContext(problem)
        r = McCormickExpressionRelaxation()

        bilinear = problem.constraint('bilinear_coef').root_expr
        ctx.set_polynomiality(bilinear, PolynomialDegree(2))

        assert r.can_relax(problem, bilinear, ctx)

        result = r.relax(problem, bilinear, ctx)
        self._check_constraints(result)
        expr = result.expression
        assert expr.expression_type == ExpressionType.Linear
        assert np.allclose(expr.coefficient(expr.children[0]), np.array([5.0]))

    def test_bilinear_sum(self, problem):
        ctx = ProblemContext(problem)
        r = McCormickExpressionRelaxation()

        bilinear = problem.constraint('bilinear_sum').root_expr
        ctx.set_polynomiality(bilinear, PolynomialDegree(2))

        assert r.can_relax(problem, bilinear, ctx)

        result = r.relax(problem, bilinear, ctx)
        assert len(result.constraints) == 12
        expr = result.expression
        assert expr.expression_type == ExpressionType.Linear
        assert len(expr.children) == 3


    @pytest.mark.skip('Requires better conversion to Quadratic')
    def test_bilinear_with_coef_2(self, problem):
        ctx = ProblemContext(problem)
        r = McCormickExpressionRelaxation()

        bilinear = problem.constraint('bilinear_coef_2').root_expr
        ctx.set_polynomiality(bilinear, PolynomialDegree(2))

        assert r.can_relax(problem, bilinear, ctx)

        result = r.relax(problem, bilinear, ctx)
        self._check_constraints(result)
        expr = result.expression
        assert expr.expression_type == ExpressionType.Linear
        assert np.allclose(expr.coefficient(expr.children[0]), np.array([6.0]))

    def _check_constraints(self, result):
        assert len(result.constraints) == 4

        # look for w >= x^l y + y^l x - x^l y^l
        #      and w >= x^u y + y^u x - x^U y^u
        #      and w <= x^U y + y^L x - x^U y^L
        #      and w <= x^L y + y^U x - x^L y^U
        upper_bound_constraints_count = 0
        for constraint in result.constraints:
            if constraint.lower_bound is None and constraint.upper_bound == 0:
                upper_bound_constraints_count += 1
                self._check_underestimator_expr(constraint.root_expr)

        assert upper_bound_constraints_count == 4

    def test_trilinear_terms(self, problem):
        ctx = ProblemContext(problem)
        r = McCormickExpressionRelaxation()

        trilinear = problem.constraint('trilinear').root_expr
        ctx.set_polynomiality(trilinear, PolynomialDegree(3))

        assert not r.can_relax(problem, trilinear, ctx)

    def test_power(self, problem):
        ctx = ProblemContext(problem)
        r = McCormickExpressionRelaxation()

        power = problem.constraint('power').root_expr
        ctx.set_polynomiality(power, PolynomialDegree(2))

        assert r.can_relax(problem, power, ctx)

    def _check_underestimator_expr(self, expr):
        assert len(expr.children) == 3

        new_variable_count = 0
        for ch in expr.children:
            if ch.graph is None:
                new_variable_count += 1
                assert (ch.name == '_aux_bilinear_x_y' or ch.name == '_aux_bilinear_y_x')

        assert new_variable_count == 1

        coef_prod = np.prod([expr.coefficient(v) for v in expr.children])
        assert np.isclose(coef_prod, expr.constant_term)
