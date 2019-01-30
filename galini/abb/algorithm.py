# Copyright 2018 Francesco Ceccon
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Alpha BB algorithm."""
import numpy as np
import datetime
import pytimeparse
import heapq
from galini.logging import Logger
from galini.bab import BabTree, KSectionBranchingStrategy, NodeSolution
from galini.abb.relaxation import AlphaBBRelaxation


class NodeSelectionStrategy(object):
    def __init__(self):
        self.nodes = []

    def insert_node(self, node):
        lower_bound = node.state.lower_bound
        heapq.heappush(self.nodes, (lower_bound, node))

    def next_node(self):
        _, node = heapq.heappop(self.nodes)
        return node


class BabAlgorithm(object):
    def __init__(self):
        self._init()

    def _init(self):
        self.tolerance = 1e-5
        self.start_time = None
        self.timeout = datetime.timedelta(seconds=pytimeparse.parse('1 min'))

    def has_converged(self, state):
        return (state.upper_bound - state.lower_bound) < self.tolerance

    def has_timeout(self):
        if self.start_time is None:
            return False
        now = datetime.datetime.now()
        return (now - self.start_time) > self.timeout

    def start_now(self):
        self.start_time = datetime.datetime.now()

    def solve(self, problem, **kwargs):
        self.logger = Logger.from_kwargs(kwargs)

        branching_strategy = KSectionBranchingStrategy(2)
        node_selection_strategy = NodeSelectionStrategy()
        tree = BabTree(branching_strategy, node_selection_strategy)

        self.start_now()

        self.logger.info('Solving root problem')
        root_solution = self.solve_root_problem(problem)
        tree.add_root(problem, root_solution)
        self.logger.info('Root problem solved, tree state {}', tree.state)

        if self.has_converged(tree.state):
            # problem is convex so it has converged already
            return root_solution.solution

        while not self.has_converged(tree.state) and not self.has_timeout():
            self.logger.info('Tree state at beginning of iteration: {}', tree.state)
            current_node = tree.next_node()
            if current_node is None:
                raise RuntimeError('No more nodes to visit')

            self.logger.info(
                'Visiting node {}: state={}, solution={}',
                current_node.coordinate,
                current_node.state,
                current_node.solution,
            )

            if current_node.state.lower_bound >= tree.state.upper_bound:
                self.logger.info(
                    "Skip node because it won't improve bound: node.lower_bound={}, tree.upper_bound={}",
                    current_node.state.lower_bound,
                    tree.state.upper_bound,
                )
                continue

            node_children, branching_point = current_node.branch()
            self.logger.info('Branched at point {}', branching_point)
            for child in node_children:
                solution = self.solve_problem(child.problem)
                self.logger.info('Child {} has solution {}', child.coordinate, solution)
                tree.update_node(child, solution)
        return current_node.solution

    def solve_root_problem(self, problem):
        return self.solve_problem(problem)

    def solve_problem(self, problem):
        raise NotImplementedError()


class AlphaBBAlgorithm(BabAlgorithm):
    def __init__(self, nlp_solver, minlp_solver):
        super().__init__()
        self._nlp_solver = nlp_solver
        self._minlp_solver = minlp_solver

    def solve_problem(self, problem):
        self.logger.info('Solving problem {}', problem.name)
        relaxation = AlphaBBRelaxation()
        relaxed_problem = relaxation.relax(problem)
        solution = self._minlp_solver.solve(relaxed_problem, logger=self.logger)

        assert len(solution.objectives) == 1
        relaxed_obj_value = solution.objectives[0].value

        x_value = dict([(v.name, v.value) for v in solution.variables])
        x = [x_value[v.name] for v in problem.variables]

        sol = self._minlp_solver.solve(problem)
        obj_value = sol.objectives[0].value
        return NodeSolution(
            relaxed_obj_value,
            obj_value,
            sol,
        )
