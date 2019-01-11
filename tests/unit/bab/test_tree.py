# pylint: skip-file
import pytest
import pyomo.environ as aml
from galini.pyomo import dag_from_pyomo_model
from galini.bab.node import Node
from galini.bab.strategy import  KSectionBranchingStrategy
from galini.bab.tree import BabTree


class FakeSelectionStrategy:
    def insert_node(self, node):
        pass


@pytest.fixture()
def problem():
    m = aml.ConcreteModel()
    m.I = range(5)
    m.x = aml.Var(m.I, bounds=(-1, 2))
    m.obj = aml.Objective(expr=sum(m.x[i] for i in m.I))
    return dag_from_pyomo_model(m)


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
    # t = BabTree(problem(), None)
    t = BabTree(KSectionBranchingStrategy(), FakeSelectionStrategy())
    t.add_root(problem())
    root = t.root
    for _ in range(3):
        root.add_children()

    c = root.children[2]
    for _ in range(2):
        c.add_children()

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
