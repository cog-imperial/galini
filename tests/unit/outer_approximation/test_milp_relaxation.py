# pylint: skip-file
import pytest
import pyomo.environ as aml
import numpy as np
from galini.pyomo import problem_from_pyomo_model
from galini.core import Domain
from galini.outer_approximation.milp_relaxation import MilpRelaxation


@pytest.fixture
def problem():
    m = aml.ConcreteModel(name='test_relaxation')

    m.I = range(5)
    m.x = aml.Var(m.I, bounds=(0, 10), domain=aml.NonNegativeIntegers)
    m.y = aml.Var(m.I, bounds=(0, 10), domain=aml.Reals)

    m.obj = aml.Objective(expr=sum(m.x[i] + m.y[i] for i in m.I))
    m.c = aml.Constraint(m.I, rule=lambda m, i: aml.exp(m.y[i]) <= 0)
    return problem_from_pyomo_model(m)


@pytest.fixture
def x_k():
    return np.array([2, 3, 4, 5, 6, 9, 8, 7, 6, 5])


def test_milp_relaxation_has_same_number_of_variable_as_original_problem(problem, x_k):
    r = MilpRelaxation()
    milp_problem = r.relax(problem, x_k=x_k)
    assert len(milp_problem.x) == problem.num_variables


def test_milp_relaxation_has_alpha_variable(problem, x_k):
    r = MilpRelaxation()
    milp_problem = r.relax(problem, x_k=x_k)
    assert milp_problem.alpha is not None


def test_milp_relaxation_has_outer_approximations(problem, x_k):
    r = MilpRelaxation()
    milp_problem = r.relax(problem, x_k=x_k)
    expected_constraints = problem.num_objectives + problem.num_constraints
    assert expected_constraints == len(milp_problem.lp.constraints)


def test_milp_relaxation_adds_constraints(problem, x_k):
    r = MilpRelaxation()
    milp_problem = r.relax(problem, x_k=x_k)
    new_x_k = 1.5 * x_k
    r.update_relaxation(problem, milp_problem, x_k=new_x_k)
    expected_constraints = (problem.num_objectives + problem.num_constraints) * 2
    assert expected_constraints == len(milp_problem.lp.constraints)
