# pytest: skip-file
import pytest
import numpy as np
from tests.unit.bab.conftest import create_solution, MockNodeStorage
from galini.branch_and_bound.node import Node
from galini.branch_and_bound.branching import BranchingPoint
from tests.unit.bab.conftest import problem


class MockBranching:
    def __init__(self, points):
        self.points = points

    def branch(self, node, _tree):
        var = node.storage.problem.variable_view(2)
        return BranchingPoint(var, self.points)


class MockTree:
    pass


class TestBranching:
    def test_branch_on_single_point(self, problem):
        tree = MockTree()
        node = Node(MockNodeStorage(problem), tree, coordinate=[0])
        node.update(create_solution(0.0, 1.0))
        new_nodes, _ = node.branch(MockBranching(1.5))

        assert len(new_nodes) == 2

        expected_bounds = [(-1, 1.5), (1.5, 2)]
        self._assert_bounds_are_correct(new_nodes, expected_bounds)

    def test_branch_on_list_of_points(self, problem):
        tree = MockTree()
        node = Node(MockNodeStorage(problem), tree, coordinate=[0])
        node.update(create_solution(0.0, 1.0))
        new_nodes, _ = node.branch(MockBranching([-0.5, 0.0, 0.5, 1.0, 1.5]))
        assert len(new_nodes) == 6

        expected_bounds = [
            (-1, -0.5), (-0.5, 0.0), (0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0)
        ]
        self._assert_bounds_are_correct(new_nodes, expected_bounds)

    def _assert_bounds_are_correct(self, new_nodes, expected_bounds):
        for node, (lower, upper) in zip(new_nodes, expected_bounds):
            var = node.storage.problem.variable_view(2)
            assert np.isclose(lower, var.lower_bound())
            assert np.isclose(upper, var.upper_bound())

        for i, node in enumerate(new_nodes):
            assert node.coordinate == [0, i]
