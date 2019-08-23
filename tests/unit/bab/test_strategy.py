# pylint: skip-file
import pytest
from tests.unit.bab.conftest import create_solution, MockNodeStorage
from galini.branch_and_bound.tree import BabTree
from galini.branch_and_bound.strategy import KSectionBranchingStrategy
from tests.unit.bab.conftest import problem


class FakeSelectionStrategy:
    def insert_node(self, node):
        pass


@pytest.fixture()
def solution():
    return create_solution(0.0, 1.0)


class TestKFoldBranchingStrategy:
    def test_bisect(self, problem, solution):
        bisect_strat = KSectionBranchingStrategy()
        tree = BabTree(
            MockNodeStorage(problem),
            bisect_strat,
            FakeSelectionStrategy(),
        )
        node = tree.root
        for i in range(5):
            node.update(create_solution(0.0, 1.0))
            children, _ = node.branch(bisect_strat)
            assert len(children) == 2
            for child in children:
                assert child.variable.idx == i
            node = children[0]

    def test_ksection(self, problem, solution):
        ksection_strat = KSectionBranchingStrategy(7)
        tree = BabTree(
            MockNodeStorage(problem),
            ksection_strat,
            FakeSelectionStrategy(),
        )
        node = tree.root
        for i in range(5):
            node.update(create_solution(0.0, 1.0))
            children, _ = node.branch(ksection_strat)
            assert len(children) == 7
            for child in children:
                assert child.variable.idx == i
            node = children[0]
