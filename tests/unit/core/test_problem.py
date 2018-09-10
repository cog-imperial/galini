# pylint: skip-file
import pytest
import pyomo.environ as aml
import numpy as np
from galini.core import Problem, RootProblem, ChildProblem
from galini.pyomo import dag_from_pyomo_model


@pytest.fixture()
def problem():
    m = aml.ConcreteModel()
    m.I = range(10)
    m.x = aml.Var(m.I, bounds=(-1, 2))
    m.obj = aml.Objective(expr=sum(m.x[i] for i in m.I))
    m.cons = aml.Constraint(m.I[1:], rule=lambda m, i: aml.cos(m.x[0]) * aml.sin(m.x[i]) >= 0)
    return dag_from_pyomo_model(m)


class TestRootProblem:
    def test_sets_bounds_correctly(self, problem):
        for i in range(10):
            var = problem.variable_at_index(i)
            var.set_lower_bound(0)
            var.set_upper_bound(1)

        np.testing.assert_allclose(
            np.zeros(10),
            np.array(problem.lower_bounds),
        )

        np.testing.assert_allclose(
            np.ones(10),
            np.array(problem.upper_bounds),
        )

    def test_sets_starting_point(self, problem):
        for i in range(10):
            var = problem.variable_at_index(i)
            assert not var.has_starting_point()

        for i in range(10):
            var = problem.variable_at_index(i)
            var.set_starting_point(2.0)

        for i in range(10):
            var = problem.variable_at_index(i)
            assert var.has_starting_point()

        np.testing.assert_allclose(
            np.ones(10) * 2.0,
            problem.starting_points,
        )

    def test_sets_value(self, problem):
        for i in range(10):
            var = problem.variable_at_index(i)
            assert not var.has_value()

        for i in range(10):
            var = problem.variable_at_index(i)
            var.set_value(2.0)

        for i in range(10):
            var = problem.variable_at_index(i)
            assert var.has_value()

        np.testing.assert_allclose(
            np.ones(10) * 2.0,
            problem.values,
        )

    def test_fix(self, problem):
        for i in range(10):
            var = problem.variable_at_index(i)
            assert not var.is_fixed()

        for i in range(10):
            var = problem.variable_at_index(i)
            var.fix(2.0)

        for i in range(10):
            var = problem.variable_at_index(i)
            assert var.is_fixed()

        np.testing.assert_allclose(
            np.ones(10) * 2.0,
            problem.fixed,
        )


class TestChildProblem:
    def test_has_different_variable_bounds(self, problem):
        for i in range(10):
            var = problem.variable_at_index(i)
            assert np.isclose(-1, var.lower_bound())
            assert np.isclose(2, var.upper_bound())

        child = problem.make_child();
        for i in range(10):
            var = child.variable_at_index(i)
            var.set_lower_bound(0.0)
            var.set_upper_bound(1.0)

        for i in range(10):
            var_root = problem.variable_at_index(i)
            assert np.isclose(-1, var_root.lower_bound())
            assert np.isclose(2, var_root.upper_bound())

            var_child = child.variable_at_index(i)
            assert np.isclose(0.0, var_child.lower_bound())
            assert np.isclose(1.0, var_child.upper_bound())
