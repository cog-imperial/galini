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
"""Fix integer variables of P to x_k."""

import numpy as np
from galini.core import Domain
from galini.relaxations.relaxation import Relaxation, RelaxationResult


def _inside_bounds(value, lower_bound, upper_bound):
    greater_than = value > lower_bound or np.isclose(value, lower_bound)
    less_than = value < upper_bound or np.isclose(value, upper_bound)
    return greater_than and less_than


class FixedIntegerContinuousRelaxation(Relaxation):
    """Continuous relaxation of problem, fixing integer variables."""
    def relaxed_problem_name(self, problem):
        return problem.name + '_fixed_integer'

    def relax_objective(self, problem, objective):
        return RelaxationResult(objective)

    def relax_constraint(self, problem, constraint):
        return RelaxationResult(constraint)

    def after_relax(self, problem, relaxed_problem, **kwargs):
        self.update_relaxation(problem, relaxed_problem, **kwargs)

    def update_relaxation(self, problem, relaxed, **kwargs):
        """Update fixed integer variables to x_k."""
        x_k = kwargs.pop('x_k', None)
        if x_k is None:
            raise ValueError('Missing required kwarg "x_k"')

        if len(x_k) != problem.num_variables:
            raise ValueError('"x_k" must have same size as problem variables')

        for i, variable in enumerate(relaxed.variables):
            if variable.domain != Domain.REAL:
                # compare bounds with original problem bounds
                view = problem.variable_view(variable)
                new_value = x_k[i]
                lower_bound = view.lower_bound()
                upper_bound = view.upper_bound()

                if not _inside_bounds(new_value, lower_bound, upper_bound):
                    message = (
                        'Fixed value must be within variable bounds: ' +
                        'variable={}, value={}, bounds=[{}, {}]'
                    ).format(variable.name, new_value, lower_bound, upper_bound)
                    raise RuntimeError(message)
                relaxed.fix(variable, x_k[i])
