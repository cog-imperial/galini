# pylint: skip-file
import pytest
import pyomo.environ as aml
from galini.pyomo import dag_from_pyomo_model
from galini.core import Domain
from galini.relaxations.continuous import ContinuousRelaxation


@pytest.fixture
def problem():
    m = aml.ConcreteModel(name='test_relaxation')

    m.I = range(5)
    m.x = aml.Var(m.I, domain=aml.NonNegativeIntegers)

    m.obj = aml.Objective(expr=m.x[0])
    m.cons0 = aml.Constraint(expr=sum(m.x[i] for i in m.I) >= 0)
    m.cons1 = aml.Constraint(expr=-2.0*aml.sin(m.x[0]) + aml.cos(m.x[1]) >= 0)
    m.cons2 = aml.Constraint(expr=m.x[1] * m.x[2] >= 0)

    return dag_from_pyomo_model(m)


def test_continuous_relaxation(problem):
    r = ContinuousRelaxation()
    relaxed = r.relax(problem)

    for var in relaxed.variables:
        assert relaxed.domain(var) == Domain.REAL

    assert len(problem.expression_tree_data().vertices()) == len(relaxed.expression_tree_data().vertices())
    assert len(problem.variables) == len(relaxed.variables)
    assert len(problem.objectives) == len(relaxed.objectives)
    assert len(problem.constraints) == len(relaxed.constraints)
