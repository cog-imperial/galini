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
"""Feasibility problem of P."""
import numpy as np
from galini.core import (
    Variable,
    Domain,
    Sense,
    LinearExpression,
    SumExpression,
    Objective,
    Constraint,
)
from galini.relaxations import Relaxation, RelaxationResult


class FeasibilityProblemRelaxation(Relaxation):
    def __init__(self):
        super().__init__()
        self._u = None
        self._constraint_idx = None

    def relaxed_problem_name(self, problem):
        return problem.name + '_feasibility_problem'

    def before_relax(self, problem, relaxed_problem, **kwargs):
        def _make_variable(i):
            return Variable('u_%d' % i, None, None, Domain.REAL)
        self._u = [_make_variable(i) for i in range(problem.num_constraints)]
        self._constraint_idx = dict([(c.name, i) for i, c in enumerate(problem.constraints)])

    def after_relax(self, problem, relaxed_problem):
        self._u = None
        self._constraint_idx = None

    def relax_objective(self, problem, objective):
        coefficients = np.ones(problem.num_constraints)
        expr = LinearExpression(self._u, coefficients.tolist(), 0.0)
        new_objective = Objective('objective', expr, Sense.MINIMIZE)
        return RelaxationResult(new_objective, [])

    def relax_constraint(self, problem, constraint):
        cons_idx = self._constraint_idx[constraint.name]
        u = self._u[cons_idx]
        minus_u = LinearExpression([u], [-1], 0.0)
        new_expr = SumExpression([constraint.root_expr, minus_u])
        new_constraint = Constraint(
            constraint.name,
            new_expr,
            constraint.lower_bound,
            constraint.upper_bound
        )
        return RelaxationResult(new_constraint, [])
