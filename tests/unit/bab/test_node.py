# pytest: skip-file
import pytest
import numpy as np
import pyomo.environ as aml
from galini.pyomo import dag_from_pyomo_model
from galini.bab.node import Node, BranchingPoint


@pytest.fixture()
def problem():
    m = aml.ConcreteModel()
    m.I = range(5)
    m.x = aml.Var(m.I, bounds=(-1, 2))
    m.obj = aml.Objective(expr=sum(m.x[i] for i in m.I))
    return dag_from_pyomo_model(m)


class MockBranching:
    def __init__(self, points):
        self.points = points

    def branch(self, node, _tree):
        var = node.problem.variable_view(2)
        return BranchingPoint(var, self.points)


class MockTree:
    pass


class TestBranching:
    def test_branch_on_single_point(self, problem):
        tree = MockTree()
        node = Node(problem, tree, coordinate=[0])
        new_nodes, _ = node.branch(MockBranching(1.5))

        assert len(new_nodes) == 2

        expected_bounds = [(-1, 1.5), (1.5, 2)]
        self._assert_bounds_are_correct(new_nodes, expected_bounds)

    def test_branch_on_list_of_points(self, problem):
        tree = MockTree()
        node = Node(problem, tree, coordinate=[0])
        new_nodes, _ = node.branch(MockBranching([-0.5, 0.0, 0.5, 1.0, 1.5]))
        assert len(new_nodes) == 6

        expected_bounds = [
            (-1, -0.5), (-0.5, 0.0), (0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0)
        ]
        self._assert_bounds_are_correct(new_nodes, expected_bounds)

    def _assert_bounds_are_correct(self, new_nodes, expected_bounds):
        for node, (lower, upper) in zip(new_nodes, expected_bounds):
            var = node.problem.variable_view(2)
            assert np.isclose(lower, var.lower_bound())
            assert np.isclose(upper, var.upper_bound())

        for i, node in enumerate(new_nodes):
            assert node.coordinate == [0, i]
