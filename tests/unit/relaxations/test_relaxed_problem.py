# pylint: skip-file
import pytest
import numpy as np
import pyomo.environ as pe
from galini.core import LinearExpression, QuadraticExpression, SumExpression
from galini.pyomo import dag_from_pyomo_model
from galini.bab.relaxations import LinearRelaxation
from galini.relaxations.relaxed_problem import RelaxedProblem
from galini.util import print_problem


def test_relaxed_problem():
    m = pe.ConcreteModel()
    m.I = range(10)
    m.x = pe.Var(m.I, bounds=(0, 1))
    m.obj = pe.Objective(expr=sum(m.x[i]*m.x[i] for i in m.I))
    m.c0 = pe.Constraint(expr=sum(m.x[i]*m.x[i] for i in m.I) >= 0)

    dag = dag_from_pyomo_model(m)

    relaxed = RelaxedProblem(LinearRelaxation(), dag)

    assert len(relaxed.relaxed.constraints) == 31

    linear_constraint = LinearExpression(
        [dag.variable(i) for i in m.I],
        [i for i in m.I],
        0.0
    )
    relaxed.add_constraint('test_linear', linear_constraint, None, 0.0)
    assert len(relaxed.relaxed.constraints) == 32

    quadratic_constraint = QuadraticExpression(
        [dag.variable(0)], [dag.variable(1)], [-2.0],
    )
    relaxed.add_constraint('test_quadratic', quadratic_constraint, 0.0, 0.0)
    assert len(relaxed.relaxed.constraints) == 32 + 1 + 4

    relaxed.add_constraint('test_mixed', SumExpression([linear_constraint, quadratic_constraint]), 0.0, None)
    assert len(relaxed.relaxed.constraints) == 38
