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
"""MIP solvers solution classes."""
import pulp
from galini.solvers import Status, Solution


class PulpStatus(Status):
    """Pulp solver status wrapper."""
    def __init__(self, inner):
        self._inner = inner

    def is_success(self):
        return self._inner == pulp.LpStatusOptimal

    def is_infeasible(self):
        return self._inner == pulp.LpStatusInfeasible

    def is_unbounded(self):
        return self._inner == pulp.LpStatusUnbounded

    def description(self):
        return pulp.LpStatus[self._inner]

    def __str__(self):
        return 'PulpStatus({})'.format(self.description())

    def __repr__(self):
        return '<{} at {}>'.format(str(self), hex(id(self)))


class MIPSolution(Solution):
    """Solution of a MIP problem."""
    def __init__(self, status, optimal_obj=None, optimal_vars=None,
                 dual_values=None):
        super().__init__(status, optimal_obj, optimal_vars)
        self.dual_values = dual_values
