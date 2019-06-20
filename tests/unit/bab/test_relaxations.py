# pylint: skip-file
import pytest
import numpy as np
import pyomo.environ as pe
from galini.core import LinearExpression, QuadraticExpression, SumExpression
from galini.pyomo import dag_from_pyomo_model
from galini.bab.relaxations import ConvexRelaxation, LinearRelaxation
from galini.util import print_problem


def test_convex_relaxation_of_linear_problem():
    m = pe.ConcreteModel()
    m.I = range(10)
    m.x = pe.Var(m.I)
    m.obj = pe.Objective(expr=pe.quicksum(m.x[i] for i in m.I) + 2.0)
    m.c0 = pe.Constraint(expr=pe.quicksum(2 * m.x[i] for i in m.I) - 4.0 >= 0)
    dag = dag_from_pyomo_model(m)

    relaxation = ConvexRelaxation()
    relaxed = relaxation.relax(dag)

    assert len(relaxed.objectives) == 1
    assert len(relaxed.constraints) == 1

    objective = relaxed.objectives[0]
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
    dag = dag_from_pyomo_model(m)

    relaxation = ConvexRelaxation()
    relaxed = relaxation.relax(dag)

    assert len(relaxed.objectives) == 1
    assert len(relaxed.constraints) == 1

    objective = relaxed.objectives[0]
    constraint = relaxed.constraints[0]

    assert isinstance(objective.root_expr, QuadraticExpression)
    assert isinstance(constraint.root_expr, QuadraticExpression)

    assert len(objective.root_expr.terms) == 10
    assert len(constraint.root_expr.terms) == 10


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
    dag = dag_from_pyomo_model(m)

    relaxation = ConvexRelaxation()
    relaxed = relaxation.relax(dag)

    assert len(relaxed.objectives) == 1
    assert len(relaxed.constraints) == 1

    objective = relaxed.objectives[0]
    constraint = relaxed.constraints[0]

    assert isinstance(objective.root_expr, SumExpression)
    assert isinstance(constraint.root_expr, SumExpression)

    if isinstance(objective.root_expr.children[0], QuadraticExpression):
        q, l = objective.root_expr.children
    else:
        l, q = objective.root_expr.children

    assert isinstance(q, QuadraticExpression)
    assert isinstance(l, LinearExpression)

    if isinstance(constraint.root_expr.children[0], QuadraticExpression):
        q, l = constraint.root_expr.children
    else:
        l, q = constraint.root_expr.children

    assert isinstance(q, QuadraticExpression)
    assert isinstance(l, LinearExpression)


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
    dag = dag_from_pyomo_model(m)

    relaxation = LinearRelaxation()
    relaxed = relaxation.relax(dag)
    print_problem(relaxed)
    assert len(relaxed.objectives) == 1
    # 1 objective, 1 c0, 3 * 10 x^2
    assert len(relaxed.constraints) == 1 + 1 + 3*10

    objective = relaxed.objectives[0]
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
