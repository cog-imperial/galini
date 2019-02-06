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
"""Base class for B&B algorithms."""
import heapq
import abc
from galini.logging import Logger
from galini.quantities import relative_gap
from galini.bab.strategy import KSectionBranchingStrategy
from galini.bab.tree import BabTree


class NodeSelectionStrategy(object):
    def __init__(self):
        self.nodes = []

    def insert_node(self, node):
        lower_bound = node.state.lower_bound
        heapq.heappush(self.nodes, (lower_bound, node))

    def next_node(self):
        _, node = heapq.heappop(self.nodes)
        return node


class BabAlgorithm(metaclass=abc.ABCMeta):
    def initialize(self, config):
        self.tolerance = 1e-5

    def has_converged(self, state):
        assert (state.upper_bound - state.lower_bound) > -1e-5
        return relative_gap(state.lower_bound, state.upper_bound) < self.tolerance

    def solve(self, problem, **kwargs):
        self.logger = Logger.from_kwargs(kwargs)

        branching_strategy = KSectionBranchingStrategy(2)
        node_selection_strategy = NodeSelectionStrategy()
        tree = BabTree(branching_strategy, node_selection_strategy)

        self.logger.info('Solving root problem')
        root_solution = self.solve_root_problem(problem)
        tree.add_root(problem, root_solution)
        self.logger.info('Root problem solved, tree state {}', tree.state)
        self.logger.log_add_bab_node(
            coordinate=[0],
            lower_bound=root_solution.lower_bound,
            upper_bound=root_solution.upper_bound,
        )

        if self.has_converged(tree.state):
            # problem is convex so it has converged already
            return root_solution.solution

        while not self.has_converged(tree.state):
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
                self.logger.log_prune_bab_node(current_node.coordinate)
                continue

            node_children, branching_point = current_node.branch()
            self.logger.info('Branched at point {}', branching_point)
            for child in node_children:
                solution = self.solve_problem(child.problem)
                self.logger.info('Child {} has solution {}', child.coordinate, solution)
                tree.update_node(child, solution)
                var_view = child.problem.variable_view(child.variable)
                self.logger.log_add_bab_node(
                    coordinate=child.coordinate,
                    lower_bound=solution.lower_bound,
                    upper_bound=solution.upper_bound,
                    branching_variables=[(child.variable.name, var_view.lower_bound(), var_view.upper_bound())],
                )
        return current_node.solution

    def solve_root_problem(self, problem):
        return self.solve_problem(problem)

    @abc.abstractmethod
    def solve_problem(self, problem):
        raise NotImplementedError()
