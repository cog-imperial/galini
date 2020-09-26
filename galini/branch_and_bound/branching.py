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

"""Branch & Bound branching."""
import copy

import numpy as np
import pyomo.environ as pe

from galini.math import almost_ge, almost_le


class BranchingPoint:
    def __init__(self, variable, points):
        self.variable = variable
        if not isinstance(points, list):
            points = [points]
        self.points = points

    def __str__(self):
        return 'BranchingPoint(variable={}, points={})'.format(
            self.variable.name, self.points
        )


def branch_at_point(model, current_bounds, branching_point, mc):
    """Branch problem at branching_point, returning a list of child problems."""
    current_bounds = pe.ComponentMap()
    for var in model.component_data_objects(pe.Var, active=True, descend_into=True):
        var_lb = var.lb
        var_ub = var.ub
        if var_lb is None:
            var_lb = -np.inf
        if var_ub is None:
            var_ub = np.inf
        current_bounds[var] = (var_lb, var_ub)

    var = branching_point.variable
    var_lb, var_ub = current_bounds[var]

    epsilon = mc.epsilon
    for point in branching_point.points:
        is_less_than_ub = almost_le(point, var_ub, atol=epsilon)
        is_greater_than_lb = almost_ge(point, var_lb, atol=epsilon)
        if not is_less_than_ub or not is_greater_than_lb:
            raise RuntimeError(
                'Branching outside variable bounds: {} in [{}, {}], branching at {}'.format(
                    var.name, var.lb, var.ub, point
            ))

    children = []
    new_upper_bound = var_lb
    is_integer = var.is_integer() or var.is_binary()
    for point in branching_point.points:
        new_lower_bound = new_upper_bound
        new_upper_bound = point
        var_lower_bound = \
            np.ceil(new_lower_bound) if is_integer else new_lower_bound
        var_upper_bound = \
            np.floor(new_upper_bound) if is_integer else new_upper_bound
        child = copy.copy(current_bounds)
        child[var] = (var_lower_bound, var_upper_bound)
        children.append(child)

    var_lower_bound = \
        np.ceil(new_upper_bound) if is_integer else new_upper_bound
    var_upper_bound = \
        np.floor(var_ub) if is_integer else var_ub
    child = copy.copy(current_bounds)
    child[var] = (var_lower_bound, var_upper_bound)
    children.append(child)
    return children
