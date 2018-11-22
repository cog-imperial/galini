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
from galini.core import Objective, Constraint
from galini.relaxations import Relaxation, RelaxationResult
from galini.abb.underestimator import AlphaBBUnderestimator
from galini.special_structure import detect_special_structure
from galini.underestimators import (
    McCormickUnderestimator,
    LinearUnderestimator,
    UnivariateConcaveUnderestimator,
    SumOfUnderestimators,
)

class AlphaBBRelaxation(Relaxation):
    def __init__(self):
        super().__init__()
        self._ctx = None
        self._underestimator = SumOfUnderestimators([
            LinearUnderestimator(),
            McCormickUnderestimator(),
            UnivariateConcaveUnderestimator(),
            AlphaBBUnderestimator(),
        ])

    def relaxed_problem_name(self, problem):
        return problem.name + '_alphabb'

    def before_relax(self, problem):
        ctx = detect_special_structure(problem)
        self._ctx = ctx
        pass

    def after_relax(self, problem, relaxed_problem):
        self._ctx = None
        pass

    def relax_objective(self, problem, objective):
        result = self.relax_expression(problem, objective.root_expr)
        new_objective = Objective(objective.name, result.expression, objective.sense)
        return RelaxationResult(new_objective, result.constraints)

    def relax_constraint(self, problem, constraint):
        result = self.relax_expression(problem, constraint.root_expr)
        new_constraint = Constraint(
            constraint.name, result.expression, constraint.lower_bound, constraint.upper_bound
        )
        return RelaxationResult(new_constraint, result.constraints)

    def relax_expression(self, problem, expr):
        assert self._underestimator.can_underestimate(problem, expr, self._ctx)
        result = self._underestimator.underestimate(problem, expr, self._ctx)
        return result
