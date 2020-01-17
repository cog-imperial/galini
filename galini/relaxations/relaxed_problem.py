# Copyright 2019 Francesco Ceccon
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

"""Helper class to keep a problem and its relaxation in sync."""
from galini.core import Constraint


class RelaxedProblem:
    def __init__(self, relaxation, original_problem, **kwargs):
        self.relaxation = relaxation
        self.original = original_problem
        self.relaxed = relaxation.relax(original_problem, **kwargs)

    def add_constraint(self, name, expr, lower_bound, upper_bound):
        constraint = Constraint(name, expr, lower_bound, upper_bound)
        relaxed_constraint = self.relaxation._relax_constraint(
            self.original,
            self.relaxed,
            constraint,
        )
        relaxed_constraint.metadata = constraint.metadata
        return relaxed_constraint
