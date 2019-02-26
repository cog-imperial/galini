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

"""Branch & Bound branching strategies."""
import numpy as np
from galini.bab.node import BranchingPoint


class BranchingStrategy(object):
    def branch(self, node, tree):
        pass


def range_ratio(problem, root_problem):
    # TODO: handle infinity better
    assert problem.num_variables == root_problem.num_variables
    lower_bounds = np.array(problem.lower_bounds)
    upper_bounds = np.array(problem.upper_bounds)
    root_lower_bounds = np.array(root_problem.lower_bounds)
    root_upper_bounds = np.array(root_problem.upper_bounds)
    denominator = np.abs(root_upper_bounds - root_lower_bounds) + 1e-5
    numerator = upper_bounds - lower_bounds
    finite_numerator = np.ones(problem.num_variables) * (numerator <= 1e19) * numerator
    finite_denominator = np.ones(problem.num_variables) * (denominator <= 1e19) * denominator
    return np.nan_to_num(finite_numerator / finite_denominator)


def least_reduced_variable(problem, root_problem):
    assert problem.num_variables == root_problem.num_variables
    r = range_ratio(problem, root_problem)
    var_idx = np.argmax(r)
    return problem.variable_view(var_idx)


class KSectionBranchingStrategy(BranchingStrategy):
    def __init__(self, k=2):
        if k < 2:
            raise ValueError('K must be >= 2')
        self.k = k

    def branch(self, node, tree):
        root_problem = tree.root.problem
        var = least_reduced_variable(node.problem, root_problem)
        return self._branch_on_var(var)

    def _branch_on_var(self, var):
        lower_bound = var.lower_bound()
        upper_bound = var.upper_bound()
        step = (upper_bound - lower_bound) / self.k
        points = step * (np.arange(self.k-1) + 1.0) + lower_bound
        assert np.all(np.isfinite(points))
        return BranchingPoint(var, points.tolist())
