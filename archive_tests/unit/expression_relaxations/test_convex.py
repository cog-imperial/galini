# pylint: skip-file
import pytest
import numpy as np
import pyomo.environ as aml
from suspect.convexity import Convexity
from suspect.expression import ExpressionType
from galini.pyomo import problem_from_pyomo_model
from galini.suspect import ProblemContext
from galini.expression_relaxation.convex import ConvexExpressionRelaxation


@pytest.fixture
def problem():
    m = aml.ConcreteModel()

    m.x = aml.Var()
    m.y = aml.Var()
    m.z = aml.Var()

    m.obj = aml.Objective(expr=aml.exp(m.x) + aml.exp(m.y))
    m.c0 = aml.Constraint(expr=2.0 * m.x*m.x + 3.0 * m.x*m.y + 4.0 * m.y*m.y >= 0)
    return problem_from_pyomo_model(m)



class TestConvexUnderestimator:
    def test_sum_of_convex_expressions(self, problem):
        ctx = ProblemContext(problem)
        r = ConvexExpressionRelaxation()

        constraint_expr = problem.objective.root_expr
        ctx.set_convexity(constraint_expr, Convexity.Convex)
        assert r.can_relax(problem, constraint_expr, ctx)

        result = r.relax(problem, constraint_expr, ctx)
        assert len(result.expression.children) == 2
        assert result.expression.expression_type == ExpressionType.Sum
        a, b = result.expression.children

        assert a.expression_type == ExpressionType.UnaryFunction
        assert b.expression_type == ExpressionType.UnaryFunction

        assert result.constraints == []

    def test_quadratic_expression(self, problem):
        ctx = ProblemContext(problem)
        r = ConvexExpressionRelaxation()

        constraint_expr = problem.constraint('c0').root_expr
        ctx.set_convexity(constraint_expr, Convexity.Unknown)
        assert r.can_relax(problem, constraint_expr, ctx)

        result = r.relax(problem, constraint_expr, ctx)
        expr = result.expression
        assert expr.expression_type == ExpressionType.Quadratic
        assert len(expr.terms) == 2
        for term in expr.terms:
            assert term.var1 == term.var2
            assert term.coefficient == 2.0 or term.coefficient == 4.0

        assert result.constraints == []
