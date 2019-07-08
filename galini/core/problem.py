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

import warnings
import numpy as np
import galini_core as core


class VariableView:
    def __init__(self, problem, variable):
        self.problem = problem
        self.variable = variable

    @property
    def name(self):
        return self.variable.name

    @property
    def idx(self):
        return self.variable.idx

    @property
    def domain(self):
        return self.problem.domain(self.variable)

    def lower_bound(self):
        return self.problem.lower_bound(self.variable)

    def set_lower_bound(self, bound):
        return self.problem.set_lower_bound(self.variable, bound)

    def upper_bound(self):
        return self.problem.upper_bound(self.variable)

    def set_upper_bound(self, bound):
        return self.problem.set_upper_bound(self.variable, bound)

    def has_starting_point(self):
        return self.problem.has_starting_point(self.variable)

    def set_starting_point(self, value):
        return self.problem.set_starting_point(self.variable, value)

    def starting_point(self):
        return self.problem.starting_point(self.variable)

    def has_value(self):
        return self.problem.has_value(self.variable)

    def set_value(self, value):
        return self.problem.set_value(self.variable, value)

    def value(self):
        return self.problem.value(self.variable)

    def is_fixed(self):
        return self.problem.is_fixed(self.variable)

    def fix(self, value):
        return self.problem.fix(self.variable, value)

    def unfix(self):
        return self.problem.fix(self.variable)


class Constraint:
    def __init__(self, name, root_expr, lower_bound, upper_bound):
        self.name = name
        self.root_expr = root_expr
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.metadata = dict()


class Objective:
    def __init__(self, name, root_expr, original_sense):
        self.name = name
        self.root_expr = root_expr
        self.original_sense = original_sense
        self.metadata = dict()


class _ProblemBase:
    def __init__(self, name):
        self.name = name
        self.parent = None

        self._domains = []
        self._lower_bounds = []
        self._upper_bounds = []
        self._starting_points = []
        self._starting_points_mask = []
        self._values = []
        self._values_mask = []
        self._fixed_mask = []
        self.metadata = dict()

    @property
    def _graph(self):
        raise NotImplementedError('_graph')

    def variable(self, idx_or_name):
        raise NotImplementedError('variable')

    def variable_view(self, idx_or_name_or_variable):
        if isinstance(idx_or_name_or_variable, core.Variable):
            variable = idx_or_name_or_variable
        else:
            variable = self.variable(idx_or_name_or_variable)
        return VariableView(self, variable)

    @property
    def lower_bounds(self):
        return np.array(self._lower_bounds)

    @property
    def upper_bounds(self):
        return np.array(self._upper_bounds)

    def domain(self, var):
        return self._variable_value(var, self._domains)

    def set_domain(self, var, domain):
        return self._set_variable_value(var, domain, self._domains)

    def lower_bound(self, var):
        return self._variable_value(var, self._lower_bounds)

    def set_lower_bound(self, var, value):
        return self._set_variable_value(var, value, self._lower_bounds)

    def upper_bound(self, var):
        return self._variable_value(var, self._upper_bounds)

    def set_upper_bound(self, var, value):
        return self._set_variable_value(var, value, self._upper_bounds)

    def has_starting_point(self, var):
        return self._variable_value(var, self._starting_points_mask)

    def set_starting_point(self, var, value):
        return self._set_variable_value(var, value, self._starting_points, self._starting_points_mask)

    def starting_point(self, var):
        return self._variable_value(var, self._starting_points, self._starting_points_mask)

    def has_value(self, var):
        return self._variable_value(var, self._values_mask)

    def value(self, var):
        result = self._variable_value(var, self._values, self._values_mask)
        if result is not None:
            return result
        # Variable could be fixed
        return self._variable_value(var, self._values, self._fixed_mask)

    def set_value(self, var, value):
        return self._set_variable_value(var, value, self._values, self._values_mask)

    def is_fixed(self, var):
        return self._variable_value(var, self._fixed_mask)

    def fix(self, var, value):
        return self._set_variable_value(var, value, self._values, self._fixed_mask)

    def unfix(self, var):
        self._fixed_mask[var.idx] = False

    def _variable_value(self, var, arr, mask=None):
        if not mask:
            return arr[var.idx]
        if mask[var.idx]:
            return arr[var.idx]
        return None

    def _set_variable_value(self, var, value, arr, mask=None):
        arr[var.idx] = value
        if mask:
            mask[var.idx] = True

    @property
    def size(self):
        return len(self._graph)

    @property
    def vertices(self):
        return self._graph

    def expression_tree_data(self):
        return self._graph.expression_tree_data()


def _make_relaxed(parent, name, relaxation=None):
    relaxed = RelaxedProblem(name, parent)
    lb = parent.lower_bound
    ub = parent.upper_bound
    domain = parent.domain

    # copy variables
    for var in parent.variables:
        if isinstance(var, core.Variable):
            new_var = core.Variable(
                var.name,
                lb(var),
                ub(var),
                domain(var),
            )
        else:
            new_var = core.Variable(
                var.name,
                lb(var),
                ub(var),
                domain(var),
            )
            new_var.reference = var.reference
        relaxed.add_variable(new_var)
    return relaxed


class Problem(_ProblemBase):
    def __init__(self, name):
        super().__init__(name)
        self._dag = core.Graph()

        self.name = name
        self.metadata = dict()

        self._variables = []
        self._variables_map = dict()
        self._objectives = []
        self._objectives_map = dict()

        self._constraints = []
        self._constraints_map = dict()

    def variable(self, idx_or_name):
        """Access variable by index or name."""
        if isinstance(idx_or_name, str):
            return self._variables_map[idx_or_name]
        if isinstance(idx_or_name, core.Variable):
            idx_or_name = idx_or_name.idx
        if isinstance(idx_or_name, VariableView):
            idx_or_name = idx_or_name.idx
        idx = idx_or_name
        if idx >= len(self._variables):
            raise IndexError('variable index out of range')
        return self._variables[idx]

    @property
    def variables(self):
        return self._variables

    @property
    def num_variables(self):
        return len(self._variables)

    def constraint(self, idx_or_name):
        if isinstance(idx_or_name, str):
            return self._constraints_map[idx_or_name]
        idx = idx_or_name
        if idx >= self.num_constraints:
            raise IndexError('constraint index out of range')
        return self._constraints[idx]

    @property
    def constraints(self):
        return self._constraints

    @property
    def num_constraints(self):
        return len(self._constraints)

    @property
    def objective(self):
        if self._objectives:
            return self._objectives[0]
        return None

    @property
    def objectives(self):
        warnings.warn('Problem.objectives is deprecated.', DeprecationWarning)
        return self._objectives

    @property
    def num_objectives(self):
        return len(self._objectives)

    def add_variable(self, variable):
        """Add variable to the problem."""
        if not isinstance(variable, core.Variable):
            raise ValueError('variable must be Variable')
        return self._add_variable(variable)

    def _add_variable(self, variable):
        if variable.name in self._variables_map:
            raise RuntimeError('Duplicate variable {}'.format(variable.name))

        new_var = self._dag.insert_vertex(variable)
        self._variables.append(new_var)

        self._domains.append(new_var.domain)
        self._lower_bounds.append(new_var.lower_bound)
        self._upper_bounds.append(new_var.upper_bound)
        self._starting_points.append(0.0)
        self._starting_points_mask.append(False)
        self._values.append(0.0)
        self._values_mask.append(False)
        self._fixed_mask.append(False)

        self._variables_map[new_var.name] = new_var
        return new_var

    def add_constraint(self, constraint):
        """Add constraint to the problem."""
        if constraint.name in self._constraints_map:
            raise RuntimeError('Duplicate constraint {}'.format(constraint.name))
        root_expr = self._dag.insert_tree(constraint.root_expr)
        self._constraints.append(constraint)
        self._constraints_map[constraint.name] = constraint
        return constraint

    def add_objective(self, objective):
        """Add objective to the problem."""
        if self._objectives:
            raise RuntimeError('Adding additional objectives.')
        root_expr = self._dag.insert_tree(objective.root_expr)

        self._objectives.append(objective)
        self._objectives_map[objective.name] = objective

    def make_child(self):
        return ChildProblem(self)

    def make_relaxed(self, name, relaxation=None):
        return _make_relaxed(self, name, relaxation)

    @property
    def _graph(self):
        return self._dag


class ChildProblem(_ProblemBase):
    def __init__(self, parent):
        super().__init__(parent.name)
        self._copy_variable_info(parent)
        self.parent = parent

    def _copy_variable_info(self, parent):
        self._domains = parent._domains.copy()
        self._lower_bounds = parent._lower_bounds.copy()
        self._upper_bounds = parent._upper_bounds.copy()
        self._starting_points = parent._starting_points.copy()
        self._starting_points_mask = parent._starting_points_mask.copy()
        self._values = parent._values.copy()
        self._values_mask = parent._values_mask.copy()
        self._fixed_mask = parent._fixed_mask.copy()

    def variable(self, idx_or_name):
        return self.parent.variable(idx_or_name)

    @property
    def variables(self):
        return self.parent.variables

    @property
    def num_variables(self):
        return self.parent.num_variables

    def constraint(self, name):
        return self.parent.constraint(name)

    @property
    def constraints(self):
        return self.parent.constraints

    @property
    def num_constraints(self):
        return self.parent.num_constraints

    @property
    def objective(self):
        return self.parent.objective

    @property
    def objectives(self):
        return self.parent.objectives

    @property
    def num_objectives(self):
        return self.parent.num_objectives

    def make_child(self):
        return ChildProblem(self)

    def make_relaxed(self, name, relaxation=None):
        return _make_relaxed(self, name, relaxation)

    @property
    def _graph(self):
        return self.parent._graph


class RelaxedProblem(Problem):
    def __init__(self, name, parent):
        super().__init__(name)
        self.parent = parent
