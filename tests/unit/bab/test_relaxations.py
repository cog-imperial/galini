# pylint: skip-file
import pytest
import numpy as np
import pyomo.environ as pe
from galini.core import LinearExpression, QuadraticExpression, SumExpression
from galini.pyomo import problem_from_pyomo_model
from galini.branch_and_bound.relaxations import ConvexRelaxation, LinearRelaxation
from galini.special_structure import propagate_special_structure, perform_fbbt
from galini.util import print_problem, expr_to_str


def _convex_relaxation(problem):
    bounds = perform_fbbt(
        problem,
        maxiter=10,
        timelimit=60,
    )

    bounds, monotonicity, convexity = \
        propagate_special_structure(problem, bounds)
    return ConvexRelaxation(problem, bounds, monotonicity, convexity)


def _linear_relaxation(problem):
    bounds = perform_fbbt(
        problem,
        maxiter=10,
        timelimit=60,
    )

    bounds, monotonicity, convexity = \
        propagate_special_structure(problem, bounds)
    return LinearRelaxation(problem, bounds, monotonicity, convexity)


def test_convex_relaxation_of_linear_problem():
    m = pe.ConcreteModel()
    m.I = range(10)
    m.x = pe.Var(m.I)
    m.obj = pe.Objective(expr=pe.quicksum(m.x[i] for i in m.I) + 2.0)
    m.c0 = pe.Constraint(expr=pe.quicksum(2 * m.x[i] for i in m.I) - 4.0 >= 0)
    dag = problem_from_pyomo_model(m)

    relaxation = _convex_relaxation(dag)
    relaxed = relaxation.relax(dag)

    assert relaxed.objective
    assert len(relaxed.constraints) == 1

    objective = relaxed.objective
    constraint = relaxed.constraints[0]

    assert isinstance(objective.root_expr, LinearExpression)
    assert len(objective.root_expr.children) == 10
    assert np.isclose(objective.root_expr.constant_term, 2.0)

    assert isinstance(constraint.root_expr, LinearExpression)
    assert len(constraint.root_expr.children) == 10
    assert np.isclose(constraint.root_expr.constant_term, -4.0)


def test_convex_relaxation_with_quadratic_only():
    m = pe.ConcreteModel()
    m.I = range(10)
    m.x = pe.Var(m.I)
    m.obj = pe.Objective(expr=pe.quicksum(m.x[i]*m.x[i] for i in m.I))
    m.c0 = pe.Constraint(expr=pe.quicksum(2 * m.x[i] * m.x[i] for i in m.I) >= 0)
    dag = problem_from_pyomo_model(m)

    relaxation = _convex_relaxation(dag)
    relaxed = relaxation.relax(dag)

    assert relaxed.objective
    # original constraint + 10 for disaggregated squares
    assert len(relaxed.constraints) == 1 + 10

    objective = relaxed.objective
    constraint = relaxed.constraints[0]

    assert isinstance(objective.root_expr, LinearExpression)
    assert len(objective.root_expr.children) == 10

    for constraint in relaxed.constraints[:-1]:
        assert isinstance(constraint.root_expr, SumExpression)
        children = constraint.root_expr.children
        assert len(children) == 2
        assert isinstance(children[0], LinearExpression)
        assert isinstance(children[1], QuadraticExpression)

    constraint = relaxed.constraints[-1]
    assert len(constraint.root_expr.children) == 10
    assert isinstance(constraint.root_expr, LinearExpression)

def test_convex_relaxation_with_quadratic_and_linear():
    m = pe.ConcreteModel()
    m.I = range(10)
    m.x = pe.Var(m.I)
    m.obj = pe.Objective(
        expr=pe.quicksum(m.x[i]*m.x[i] for i in m.I) + pe.quicksum(m.x[i] for i in m.I)
    )
    m.c0 = pe.Constraint(
        expr=pe.quicksum(2 * m.x[i] * m.x[i] for i in m.I) + pe.quicksum(m.x[i] for i in m.I) >= 0
    )
    dag = problem_from_pyomo_model(m)

    relaxation = _convex_relaxation(dag)
    relaxed = relaxation.relax(dag)

    print_problem(relaxed)

    assert relaxed.objective
    assert len(relaxed.constraints) == 1 + 10

    objective = relaxed.objective
    assert isinstance(objective.root_expr, SumExpression)

    assert all(
        isinstance(c, LinearExpression)
        for c in objective.root_expr.children
    )

    for constraint in relaxed.constraints[:-1]:
        assert isinstance(constraint.root_expr, SumExpression)
        children = constraint.root_expr.children
        assert len(children) == 2
        assert isinstance(children[0], LinearExpression)
        assert isinstance(children[1], QuadraticExpression)

    constraint = relaxed.constraints[-1]
    assert all(
        isinstance(c, LinearExpression)
        for c in constraint.root_expr.children
    )


def test_linear_relaxation_with_quadratic_and_linear():
    m = pe.ConcreteModel()
    m.I = range(10)
    m.x = pe.Var(m.I, bounds=(0, 1))
    m.obj = pe.Objective(
        expr=pe.quicksum(m.x[i]*m.x[i] for i in m.I) + pe.quicksum(m.x[i] for i in m.I)
    )
    m.c0 = pe.Constraint(
        expr=pe.quicksum(2 * m.x[i] * m.x[i] for i in m.I) + pe.quicksum(m.x[i] for i in m.I) >= 0
    )
    dag = problem_from_pyomo_model(m)

    relaxation = _linear_relaxation(dag)
    relaxed = relaxation.relax(dag)
    print_problem(relaxed)
    assert relaxed.objective
    # 1 objective, 1 c0, 4 * 10 x^2 (3 mccormick, 1 midpoint)
    assert len(relaxed.constraints) == 1 + 1 + 4*10

    objective = relaxed.objective
    constraint = relaxed.constraint('c0')

    assert isinstance(objective.root_expr, LinearExpression)
    assert isinstance(constraint.root_expr, SumExpression)

    # Root is only objvar
    assert len(objective.root_expr.children) == 1

    c0, c1 = constraint.root_expr.children

    assert isinstance(c0, LinearExpression)
    assert isinstance(c1, LinearExpression)
    assert len(c0.children) == 10
    assert len(c1.children) == 10
