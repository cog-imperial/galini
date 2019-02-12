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

"""A relaxation that removes integrality constraints on variables."""
from galini.core import Variable, Domain
from galini.relaxations.relaxation import Relaxation, RelaxationResult



class ContinuousRelaxation(Relaxation):
    def relaxed_problem_name(self, problem):
        return problem.name + '_continuous'

    def relax_variable(self, problem, variable):
        return Variable(
            variable.name,
            problem.lower_bound(variable),
            problem.upper_bound(variable),
            Domain.REAL,
        )

    def relax_objective(self, problem, objective):
        return RelaxationResult(objective)

    def relax_constraint(self, problem, constraint):
        return RelaxationResult(constraint)
