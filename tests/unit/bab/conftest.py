# pylint: skip-file
from galini.solvers.solution import OptimalObjective, Solution
from galini.bab.node import NodeSolution


def create_solution(lb, ub):
    return NodeSolution(MockSolution(lb), MockSolution(ub))


class MockStatus(object):
    def is_success(self):
        return True

    def description(self):
        return 'Success'


class MockSolution(Solution):
    def __init__(self, obj):
        self.status = MockStatus()
        self.objective = OptimalObjective(name='obj', value=obj)
        self.variables = []


class MockSelectionStrategy:
    def insert_node(self, node):
        pass
