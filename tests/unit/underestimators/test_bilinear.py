# pylint: skip-file
import pytest
import numpy as np
import pyomo.environ as aml
from suspect.polynomial import PolynomialDegree
from suspect.expression import ExpressionType
from galini.pyomo import dag_from_pyomo_model
from galini.suspect import ProblemContext
from galini.underestimators.bilinear import McCormickUnderestimator


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
    m.trilinear = aml.Constraint(expr=m.x*m.y*m.z >= 0)
    m.power = aml.Constraint(expr=m.x**2 >= 0)

    return dag_from_pyomo_model(m)


class TestMcCormickUnderestimator:
    def test_bilinear_terms(self, problem):
        ctx = ProblemContext(problem)
        r = McCormickUnderestimator()

        bilinear = problem.constraint('bilinear').root_expr
        ctx.set_polynomiality(bilinear, PolynomialDegree(2))

        assert r.can_underestimate(problem, bilinear, ctx)

        result = r.underestimate(problem, bilinear, ctx)
        self._check_constraints(result)
        assert result.expression.expression_type == ExpressionType.Variable

    def test_bilinear_with_coef(self, problem):
        ctx = ProblemContext(problem)
        r = McCormickUnderestimator()

        bilinear = problem.constraint('bilinear_coef').root_expr
        ctx.set_polynomiality(bilinear, PolynomialDegree(2))

        assert r.can_underestimate(problem, bilinear, ctx)

        result = r.underestimate(problem, bilinear, ctx)
        self._check_constraints(result)
        expr = result.expression
        assert expr.expression_type == ExpressionType.Linear
        assert np.allclose(expr.coefficients, np.array([5.0]))

    def test_bilinear_with_coef_2(self, problem):
        ctx = ProblemContext(problem)
        r = McCormickUnderestimator()

        bilinear = problem.constraint('bilinear_coef_2').root_expr
        ctx.set_polynomiality(bilinear, PolynomialDegree(2))

        assert r.can_underestimate(problem, bilinear, ctx)

        result = r.underestimate(problem, bilinear, ctx)
        self._check_constraints(result)
        expr = result.expression
        assert expr.expression_type == ExpressionType.Linear
        assert np.allclose(expr.coefficients, np.array([6.0]))

    def _check_constraints(self, result):
        assert len(result.constraints) == 4

        # look for w >= x^l y + y^l x - x^l y^l
        #      and w >= x^u y + y^u x - x^U y^u
        upper_bound_constraints_count = 0
        for constraint in result.constraints:
            if constraint.lower_bound is None and constraint.upper_bound == 0:
                upper_bound_constraints_count += 1
                self._check_underestimator_expr(constraint.root_expr)

        assert upper_bound_constraints_count == 2

        # look for w <= x^U y + y^L x - x^U y^L
        #      and w <= x^L y + y^U x - x^L y^U
        lower_bound_constraints_count = 0
        for constraint in result.constraints:
            if constraint.upper_bound is None and constraint.lower_bound == 0:
                lower_bound_constraints_count += 1
                self._check_underestimator_expr(constraint.root_expr)

        assert lower_bound_constraints_count == 2

    def test_trilinear_terms(self, problem):
        ctx = ProblemContext(problem)
        r = McCormickUnderestimator()

        trilinear = problem.constraint('trilinear').root_expr
        ctx.set_polynomiality(trilinear, PolynomialDegree(3))

        assert not r.can_underestimate(problem, trilinear, ctx)

    def test_power(self, problem):
        ctx = ProblemContext(problem)
        r = McCormickUnderestimator()

        power = problem.constraint('power').root_expr
        ctx.set_polynomiality(power, PolynomialDegree(2))

        assert not r.can_underestimate(problem, power, ctx)

    def _check_underestimator_expr(self, expr):
        assert len(expr.children) == 3

        new_variable_count = 0
        for ch in expr.children:
            if ch.problem is None:
                new_variable_count += 1
                assert (ch.name == '_aux_bilinear_x_y' or ch.name == '_aux_bilinear_y_x')

        assert new_variable_count == 1

        coef_prod = np.prod(expr.coefficients)
        assert np.isclose(coef_prod, expr.constant_term)
