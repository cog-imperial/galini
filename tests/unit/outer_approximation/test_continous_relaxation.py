# pylint: skip-file
import pytest
import pyomo.environ as aml
import numpy as np
from galini.pyomo import dag_from_pyomo_model
from galini.core import Domain
from galini.outer_approximation.continuous_relaxation import FixedIntegerContinuousRelaxation


@pytest.fixture
def problem():
    m = aml.ConcreteModel(name='test_relaxation')

    m.I = range(5)
    m.x = aml.Var(m.I, domain=aml.NonNegativeIntegers)
    m.y = aml.Var(m.I, domain=aml.Reals)

    m.obj = aml.Objective(expr=sum(m.x[i] + m.y[i] for i in m.I))
    return dag_from_pyomo_model(m)


def test_continous_relaxation_throws_if_x_k_not_specified(problem):
    r = FixedIntegerContinuousRelaxation()
    with pytest.raises(ValueError):
        relaxed = r.relax(problem)


def test_continous_relaxation_throws_if_x_k_wrong_size(problem):
    r = FixedIntegerContinuousRelaxation()
    x_k = np.zeros(problem.num_variables - 1)
    with pytest.raises(ValueError):
        relaxed = r.relax(problem, x_k=x_k)


def test_continuous_relaxation(problem):
    r = FixedIntegerContinuousRelaxation()
    x_k = np.arange(problem.num_variables)
    relaxed = r.relax(problem, x_k=x_k)

    for var in relaxed.variables:
        if var.name[0] == 'x':
            view = relaxed.variable_view(var)
            assert view.is_fixed()

    assert len(problem.expression_tree_data().vertices()) == len(relaxed.expression_tree_data().vertices())
    assert len(problem.variables) == len(relaxed.variables)
    assert len(problem.objectives) == len(relaxed.objectives)
    assert len(problem.constraints) == len(relaxed.constraints)


def test_continuous_relaxation_can_update_fixed_variables(problem):
    r = FixedIntegerContinuousRelaxation()
    x_k = np.arange(problem.num_variables)
    relaxed = r.relax(problem, x_k=x_k)

    for var in relaxed.variables:
        if var.name[0] == 'x':
            view = relaxed.variable_view(var)
            assert view.is_fixed()

    new_x_k = x_k * 2.0
    r.update_relaxation(problem, relaxed, x_k=new_x_k)
    for i, var in enumerate(relaxed.variables):
        if var.name[0] == 'x':
            view = relaxed.variable_view(var)
            assert view.is_fixed()
            assert new_x_k[i] == view.value()
