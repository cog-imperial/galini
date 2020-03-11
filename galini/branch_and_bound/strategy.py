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
import pyomo.environ as pe

from galini.branch_and_bound.branching import BranchingPoint
from galini.math import is_inf, is_close


class BranchingStrategy(object):
    def branch(self, node, tree):
        pass


def max_range_ratio(model, root_bounds):
    max_ratio = -np.inf
    max_var = None
    for var in model.component_data_objects(pe.Var, active=True):
        root_var_bounds = root_bounds.get(var, None)
        if root_var_bounds is None:
            pass
        (root_lb, root_ub) = root_var_bounds
        if root_lb is None or root_ub is None:
            continue
        lb = var.lb
        ub = var.ub
        if lb is None or ub is None:
            continue
        if np.isclose(root_ub, root_lb):
            continue
        var_ratio = (ub - lb) / (root_ub - root_lb)
        if var_ratio > max_ratio:
            max_ratio = var_ratio
            max_var = var
    return max_var


def least_reduced_variable(model, root_bounds):
    # If any variable is unbounded, branch at 0.0
    var = max_range_ratio(model, root_bounds)
    # Could not compute range ratio, for example all bounded variables are fixed
    # Return the first variable to be unbounded.
    if var is not None:
        return var
    for var in model.component_data_objects(pe.Var, active=True):
        if not var.has_lb() or not var.has_ub():
            return var
    return None


class KSectionBranchingStrategy(BranchingStrategy):
    def __init__(self, tolerance, user_upper_bound, user_integer_upper_bound, k=2):
        if k < 2:
            raise ValueError('K must be >= 2')
        self.k = k
        self.tolerance = tolerance
        self.user_upper_bound = user_upper_bound
        self.user_integer_upper_bound = user_integer_upper_bound

    def branch(self, node, tree):
        root_problem = tree.root.storage.branching_data()
        node_problem = node.storage.branching_data()
        var = least_reduced_variable(node_problem, root_problem)
        if var is None:
            return None
        if is_close(var.upper_bound(), var.lower_bound(), atol=self.tolerance):
            return None
        return self._branch_on_var(var)

    def _branch_on_var(self, var):
        lower_bound = var.lower_bound()
        domain = var.domain
        if is_inf(lower_bound):
            if domain.is_real():
                lower_bound = -self.user_upper_bound
            else:
                lower_bound = -self.user_integer_upper_bound

        upper_bound = var.upper_bound()
        if is_inf(upper_bound):
            if domain.is_real():
                upper_bound = self.user_upper_bound
            else:
                upper_bound = self.user_integer_upper_bound

        step = (upper_bound - lower_bound) / self.k
        points = step * (np.arange(self.k-1) + 1.0) + lower_bound
        assert np.all(np.isfinite(points))
        return BranchingPoint(var, points.tolist())
