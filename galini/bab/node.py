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
from galini.core import VariableView


class BranchingPoint(object):
    def __init__(self, variable, points):
        self.variable = variable
        if not isinstance(points, list):
            points = [points]
        self.points = points


class Node(object):
    def __init__(self, problem, tree=None, coordinate=None, variable=None, solution=None):
        self.children = None
        self.problem = problem
        self.tree = tree
        self.coordinate = coordinate
        self.variable = variable
        self.solution = solution

    def branch(self, strategy=None):
        """Branch at the current node using strategy."""
        if self.children is not None:
            raise RuntimeError('Trying to branch on node with children.')
        if strategy is None:
            if self.tree is None:
                raise RuntimeError('Trying to branch without associated strategy.')
            strategy = self.tree.branching_strategy
        branching_point = strategy.branch(self, self.tree)
        if branching_point is None:
            raise RuntimeError('Could not branch')
        self.branch_at_point(branching_point)
        return self.children

    def branch_at_point(self, branching_point):
        branching_var = branching_point.variable
        if isinstance(branching_var, VariableView):
            branching_var = branching_var.variable
        var = self.problem.variable_view(branching_var.idx)
        for point in branching_point.points:
            if point < var.lower_bound() or point > var.upper_bound():
                raise RuntimeError('Branching outside variable bounds')
        new_upper_bound = var.lower_bound()
        for point in branching_point.points:
            new_lower_bound = new_upper_bound
            new_upper_bound = point
            self._add_children_branched_at(branching_var, new_lower_bound, new_upper_bound)
        self._add_children_branched_at(branching_var, new_upper_bound, var.upper_bound())

    def _add_children_branched_at(self, branching_var, new_lower_bound, new_upper_bound):
        child_problem = self.add_children(branching_var)
        var = child_problem.variable_view(branching_var.idx)
        var.set_lower_bound(new_lower_bound)
        var.set_upper_bound(new_upper_bound)

    def add_children(self, branching_var=None):
        child = self.problem.make_child()
        if self.coordinate is not None:
            num_children = 0 if self.children is None else len(self.children)
            coordinate = self.coordinate.copy()
            coordinate.append(num_children)
        else:
            coordinate = None

        child_node = Node(child, self.tree, coordinate, branching_var)

        if self.children is None:
            self.children = [child_node]
        else:
            self.children.append(child_node)
        return child
