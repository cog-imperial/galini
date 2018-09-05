# pylint: skip-file
import pytest
import pyomo.environ as aml
from galini.core import Problem, RootProblem, ChildProblem
from galini.pyomo import dag_from_pyomo_model


@pytest.fixture()
def problem():
    m = aml.ConcreteModel()
    m.I = range(10)
    m.x = aml.Var(m.I, bounds=(-1, 2))
    m.obj = aml.Objective(expr=sum(m.x[i] for i in range(m.I)))
    m.cons = aml.Objective(m.I[1:], rule=lambda m, i: aml.cos(m.x[0]) * aml.sin(m.x[i]) >= 0)
    return dag_from_pyomo_model(m)


class RootProblemTest:
    def test_sets_bounds_correctly(self, problem):
        pass
