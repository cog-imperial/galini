# pylint: skip-file
import pytest
import numpy as np
from tests.unit.bab.conftest import (
    MockSelectionStrategy, create_solution, create_problem, MockNodeStorage
)
from galini.branch_and_bound.strategy import KSectionBranchingStrategy
from galini.branch_and_bound.branching import BranchingPoint
from galini.branch_and_bound.tree import BabTree


@pytest.fixture()
def solution():
    return create_solution(-20.0, -10.0)


@pytest.fixture()
def tree():
    """Create a BaBTree like:

            0
       +----+----+
       |    |    |
       1    2    3
               +-+-+
               |   |
               4   5

    with coordinates:

     0 : [0]
     1 : [0, 0]
     2 : [0, 1]
     3 : [0, 2]
     4 : [0, 2, 0]
     5 : [0, 2, 1]

    """
    problem = create_problem()
    storage = MockNodeStorage(problem)
    t = BabTree(storage, KSectionBranchingStrategy(), MockSelectionStrategy())
    t.update_root(create_solution(-30.0, 0.0))
    root = t.root
    bp = BranchingPoint(problem.variable(0), [0.0, 0.5])
    root.branch_at_point(bp)

    c = root.children[2]
    bp = BranchingPoint(problem.variable(1), [0.0])
    c.branch_at_point(bp)

    return t


class TestBabTreeCoordinates:
    def test_starting_with_0_end_0(self, tree):
        node = tree.node([0])
        assert node is tree.root

    def test_starting_without_0(self, tree):
        with pytest.raises(ValueError):
            node = tree.node([2, 1])

    def test_out_of_bounds(self, tree):
        with pytest.raises(IndexError):
            node = tree.node([0, 0, 1])

        with pytest.raises(IndexError):
            node = tree.node([0, 3])


class TestBabTreeState:
    def test_update_root(self):
        tree = BabTree(
            MockNodeStorage(create_problem()),
            KSectionBranchingStrategy(),
            MockSelectionStrategy()
        )
        sol = create_solution(5.0, 10.0)
        tree.update_root(sol)
        assert np.isclose(5.0, tree.lower_bound)
        assert np.isclose(10.0, tree.upper_bound)

    def test_update_with_new_lower_bound(self):
        tree = BabTree(
            MockNodeStorage(create_problem()),
            KSectionBranchingStrategy(),
            MockSelectionStrategy()
        )
        sol = create_solution(5.0, 10.0)
        tree.update_root(sol)

        root_children, _ = tree.branch_at_node(tree.root)
        a, b = root_children
        tree.update_node(a, create_solution(7.0, 11.0))

        tree.branch_at_node(a)

        assert np.isclose(5.0, tree.lower_bound)
        assert np.isclose(10.0, tree.upper_bound)

        tree.update_node(b, create_solution(6.0, 10.0))
        children, _ = tree.branch_at_node(b)

        assert np.isclose(6.0, tree.lower_bound)
        assert np.isclose(10.0, tree.upper_bound)

        a, b = children
        tree.update_node(a, create_solution(7.0, 10.0))
        tree.branch_at_node(a)

        assert np.isclose(6.0, tree.lower_bound)
        assert np.isclose(10.0, tree.upper_bound)

    def test_update_with_new_upper_bound(self):
        tree = BabTree(
            MockNodeStorage(create_problem()),
            KSectionBranchingStrategy(),
            MockSelectionStrategy()
        )
        sol = create_solution(5.0, 10.0)

        tree.update_root(sol)
        root_children, _ = tree.branch_at_node(tree.root)
        a, b = root_children

        assert np.isclose(5.0, tree.lower_bound)
        assert np.isclose(10.0, tree.upper_bound)

        tree.update_node(a, create_solution(7.0, 11.0))
        tree.branch_at_node(a)

        assert np.isclose(5.0, tree.lower_bound)
        assert np.isclose(10.0, tree.upper_bound)

        tree.update_node(b, create_solution(6.0, 9.0))
        tree.branch_at_node(b)

        assert np.isclose(6.0, tree.lower_bound)
        assert np.isclose(9.0, tree.upper_bound)

    def test_update_with_both_new_bounds(self):
        tree = BabTree(
            MockNodeStorage(create_problem()),
            KSectionBranchingStrategy(),
            MockSelectionStrategy()
        )
        sol = create_solution(5.0, 10.0)

        tree.update_root(sol)
        root_children, _ = tree.branch_at_node(tree.root)
        a, b = root_children


        assert np.isclose(5.0, tree.lower_bound)
        assert np.isclose(10.0, tree.upper_bound)

        tree.update_node(a, create_solution(7.0, 11.0))
        tree.branch_at_node(a)

        assert np.isclose(5.0, tree.lower_bound)
        assert np.isclose(10.0, tree.upper_bound)

        tree.update_node(b, create_solution(6.0, 9.0))
        children, _ = tree.branch_at_node(b)
        a, b = children

        assert np.isclose(6.0, tree.lower_bound)
        assert np.isclose(9.0, tree.upper_bound)

        tree.update_node(a, create_solution(7.0, 10.0))
        tree.branch_at_node(a)

        assert np.isclose(6.0, tree.lower_bound)
        assert np.isclose(9.0, tree.upper_bound)
