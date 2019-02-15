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

"""Branch & Bound tree."""
from collections import namedtuple
import numpy as np
from galini.bab.node import Node, NodeSolution


TreeState = namedtuple('TreeState', ['lower_bound', 'upper_bound', 'nodes_visited'])


class BabTree(object):
    def __init__(self, branching_strategy, selection_strategy):
        self.root = None
        self.branching_strategy = branching_strategy
        self.selection_strategy = selection_strategy
        self.state = TreeState(lower_bound=-np.inf, upper_bound=np.inf, nodes_visited=0)
        self.best_solution = None

    def add_root(self, problem, solution):
        self.root = Node(problem, tree=self, coordinate=[0])
        self.update_node(self.root, solution)

    def has_nodes(self):
        return self.selection_strategy.has_nodes()

    def next_node(self):
        return self.selection_strategy.next_node()

    def update_node(self, node, solution):
        assert isinstance(solution, NodeSolution)
        node.update(solution)
        self.update_state(solution)
        self.selection_strategy.insert_node(node)

    def node(self, coord):
        if not isinstance(coord, list):
            raise TypeError('BabTree coord must be a list')

        if coord[0] != 0:
            raise ValueError('First node must be root with index 0')

        if self.root is None:
            raise ValueError('Must add root node to tree.')

        coord = coord[1:]
        current = self.root
        for i, c in enumerate(coord):
            if current.children is None or c >= len(current.children):
                raise IndexError('Node index out of bounds at {}', coord[:i])
            current = current.children[c]
        return current

    def update_state(self, solution):
        # new_lower_bound = max(solution.lower_bound, self.state.lower_bound)
        new_lower_bound = self.state.lower_bound
        if solution.upper_bound < self.state.upper_bound:
            self.best_solution = solution
        new_upper_bound = min(solution.upper_bound, self.state.upper_bound)
        new_nodes_visited = self.state.nodes_visited + 1
        self.state = TreeState(new_lower_bound, new_upper_bound, new_nodes_visited)
