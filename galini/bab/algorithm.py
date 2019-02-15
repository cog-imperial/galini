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
from galini.quantities import relative_gap, absolute_gap
from galini.bab.strategy import KSectionBranchingStrategy
from galini.bab.tree import BabTree


class NodeSelectionStrategy(object):
    class _Node(object):
        def __init__(self, node):
            self.inner = node

        def __lt__(self, other):
            return self.inner.state.lower_bound < other.inner.state.lower_bound

    def __init__(self):
        self.nodes = []

    def insert_node(self, node):
        heapq.heappush(self.nodes, self._Node(node))

    def has_nodes(self):
        return len(self.nodes) > 0

    def next_node(self):
        if len(self.nodes) == 0:
            return None
        node = heapq.heappop(self.nodes)
        return node.inner


class BabAlgorithm(metaclass=abc.ABCMeta):
    def initialize(self, config):
        self.tolerance = config.get('tolerance', 1e-8)
        self.relative_tolerance = config.get('relative_tolerance', 1e-8)
        self.node_limit = config.get('node_limit', 10000000000000)

    def has_converged(self, state):
        return False
        rel_gap = relative_gap(state.lower_bound, state.upper_bound)
        abs_gap = absolute_gap(state.lower_bound, state.upper_bound)
        return (
            rel_gap <= self.relative_tolerance or
            abs_gap <= self.tolerance
        )

    def _node_limit_exceeded(self, state):
        return state.nodes_visited > self.node_limit

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

        while not self.has_converged(tree.state) and not self._node_limit_exceeded(tree.state):
            self.logger.info('Tree state at beginning of iteration: {}', tree.state)
            if not tree.has_nodes():
                return tree.best_solution.solution
            current_node = tree.next_node()

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

            if not current_node.solution.status.is_success():
                self.logger.info("Skip node because it was not feasible")
                self.logger.log_prune_bab_node(current_node.coordinate)
                continue

            node_children, branching_point = current_node.branch()
            self.logger.info('Branched at point {}', branching_point)
            for child in node_children:
                solution = self.solve_problem(child.problem)
                self.logger.info('Child {} has solution {}', child.coordinate, solution)
                tree.update_node(child, solution)
                self.logger.info('New tree state {}', tree.state)
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
