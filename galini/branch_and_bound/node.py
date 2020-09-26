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

"""Branch & Bound node."""

from collections import namedtuple

import numpy as np

NodeState = namedtuple(
    'NodeState',
    ['lower_bound_solution']
)


class NodeSolution:
    """Problem solution associated with a node."""
    def __init__(self, lower_bound_solution, upper_bound_solution):
        self.lower_bound_solution = lower_bound_solution
        self.upper_bound_solution = upper_bound_solution

    @property
    def lower_bound_success(self):
        solution = self.lower_bound_solution
        if solution is None:
            return False
        return solution.status.is_success()

    @property
    def upper_bound_success(self):
        solution = self.upper_bound_solution
        if solution is None:
            return False
        return solution.status.is_success()

    @property
    def lower_bound(self):
        solution = self.lower_bound_solution
        if solution is None:
            return -np.inf
        if not solution.status.is_success():
            return -np.inf
        return solution.best_objective_estimate()

    @property
    def upper_bound(self):
        solution = self.upper_bound_solution
        if solution is None:
            return np.inf
        if not solution.status.is_success():
            return np.inf
        return solution.objective_value()

    def __str__(self):
        return 'NodeSolution(lower_bound={}, upper_bound={})'.format(
            self.lower_bound, self.upper_bound
        )

    def __repr__(self):
        return '<{} at {}>'.format(str(self), hex(id(self)))


class Node:
    def __init__(self, storage, tree=None, parent=None,
                 coordinate=None, variable=None, solution=None):
        self.children = None
        self.storage = storage
        self.tree = tree
        self.parent = parent
        self.coordinate = coordinate
        self.variable = variable
        self.state = None
        self.initial_feasible_solution = None

        if solution:
            self.update(solution)

    def update(self, solution, can_override=False):
        if not can_override:
            assert self.state is None
        assert isinstance(solution, NodeSolution)

        if can_override and self.state is not None:
            new_lower_bound_solution = _best_solution(
                self.state.lower_bound_solution,
                solution.lower_bound_solution,
                lambda a, b: b > a
            )
        else:
            new_lower_bound_solution = solution.lower_bound_solution

        self.state = NodeState(
            lower_bound_solution=new_lower_bound_solution,
        )

    @property
    def has_solution(self):
        return self.state is not None

    @property
    def has_parent(self):
        return self.parent is not None

    @property
    def lower_bound(self):
        assert self.state is not None
        solution = self.state.lower_bound_solution
        if solution is None:
            return -np.inf
        if not solution.status.is_success():
            return -np.inf
        if self.parent:
            return max(
                solution.best_objective_estimate(),
                self.parent.lower_bound,
            )
        return solution.best_objective_estimate()

    def branch(self, mc, strategy=None):
        """Branch at the current node using strategy."""
        if self.children is not None:
            raise RuntimeError('Trying to branch on node with children.')

        if strategy is None:
            if self.tree is None:
                raise RuntimeError('Trying to branch without associated strategy.')
            strategy = self.tree.branching_strategy

        lower_bound_solution = self.state.lower_bound_solution
        if lower_bound_solution is None:
            return None, None

        if lower_bound_solution.status.is_unbounded():
            # TODO(fra): how to handle unbounded lower bounding problems?
            return None, None

        if not lower_bound_solution.status.is_success():
            return None, None

        branching_point = strategy.branch(self, self.tree)
        if branching_point is None:
            return None, None
        return self.branch_at_point(branching_point, mc)

    def branch_at_point(self, branching_point, mc):
        children_node_storage = self.storage.branch_at_point(branching_point, mc)
        for child_storage in children_node_storage:
            self.add_children(child_storage, branching_point.variable)
        return self.children, branching_point

    def add_children(self, node_storage, branching_var=None):
        if self.coordinate is not None:
            num_children = 0 if self.children is None else len(self.children)
            coordinate = self.coordinate.copy()
            coordinate.append(num_children)
        else:
            coordinate = None

        child_node = Node(
            node_storage, self.tree, self, coordinate, branching_var
        )

        if self.children is None:
            self.children = [child_node]
        else:
            self.children.append(child_node)

    @property
    def coordinate_hash(self):
        return '-'.join([str(c) for c in self.coordinate])


def _best_solution(existing_solution, new_solution, improves_solution):
    if existing_solution is None:
        return new_solution
    if not existing_solution.status.is_success():
        return new_solution
    if new_solution is None:
        return existing_solution
    if not new_solution.status.is_success():
        return existing_solution

    existing_objective = existing_solution.objective_value()
    new_objective = new_solution.objective_value()
    if improves_solution(existing_objective, new_objective):
        return new_solution

    return existing_solution