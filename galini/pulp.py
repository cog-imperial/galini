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
"""Interface GALINI with Coin-OR pulp."""
import pulp
import logging
from galini.solvers.solution import OptimalObjective, OptimalVariable, Status, Solution

log = logging.getLogger(pulp.__name__)
log.setLevel(9001) # silence pulp


class PulpStatus(Status):
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


def pulp_solve(problem):
    solver = _solver()
    status =  problem.solve(solver)
    return Solution(
        PulpStatus(status),
        [OptimalObjective(problem.objective.name, pulp.value(problem.objective))],
        [OptimalVariable(var.name, pulp.value(var)) for var in problem.variables()]
    )

def _solver():
    cplex = pulp.CPLEX_PY()
    if cplex.available():
        return cplex
    return pulp.PULP_CBC_CMD()


class CplexSolver(object):
    """Wrapper around pulp.CPLEX_PY to integrate with GALINI."""
    def __init__(self):
        self._inner = pulp.CPLEX_PY()

    def available(self):
        return self._inner.available()

    def actualSolve(self, lp):
        # Same as CPLEX_PY.actualSolve, but overrides log settings
        self._inner.buildSolverModel(lp)

        self._setup_logs()

        self._inner.callSolver(lp)
        solutionStatus = self._inner.findSolutionValues(lp)
        for var in lp.variables():
            var.modified = False
        for constraint in lp.constraints.values():
            constraint.modified = False
        return solutionStatus

    def _setup_logs(self):
        if not self.available():
            return
        model = self._inner.solverModel
