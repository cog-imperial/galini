#  Copyright 2019 Francesco Ceccon
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Branch & Cut node storage. Contains original and convex problem."""

from galini.branch_and_bound.branching import branch_at_point
from galini.cuts.pool import CutNodeStorage, CutPool

class _NodeStorageBase:
    def __init__(self, problem):
        self.problem = problem
        self._branching_var = None
        self._branching_point = None

    @property
    def is_root(self):
        pass

    def branching_data(self):
        return self.problem

    def branch_at_point(self, branching_point):
        problem_children = branch_at_point(self.problem, branching_point)

        return [NodeStorage(problem, self, branching_point.variable) for problem in problem_children]


class NodeStorage(_NodeStorageBase):
    def __init__(self, problem, parent, branching_variable):
        super().__init__(problem)
        self.cut_pool = parent.cut_pool
        self.cut_node_storage = \
            CutNodeStorage(parent.cut_node_storage, parent.cut_pool)
        self.branching_variable = branching_variable

    @property
    def is_root(self):
        return False

class RootNodeStorage(_NodeStorageBase):
    def __init__(self, problem):
        super().__init__(problem)
        self.cut_pool = CutPool(problem)
        self.cut_node_storage = CutNodeStorage(None, self.cut_pool)

    @property
    def is_root(self):
        return True
