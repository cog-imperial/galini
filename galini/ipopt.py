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
"""Solve NLP using Ipopt."""
import numpy as np
from galini.logging import Logger
from galini.solvers import (
    Solver,
    Solution,
    Status,
    OptimalObjective,
    OptimalVariable,
)
from galini.core import ipopt_solve, IpoptSolution


class IpoptStatus(Status):
    def __init__(self, status):
        self._status = status

    def is_success(self):
        return self._status == IpoptSolution.StatusType.success

    def is_infeasible(self):
        return self._status == IpoptSolution.StatusType.local_infeasibility

    def is_unbounded(self):
        return False

    def description(self):
        return str(self._status)


class IpoptNLPSolver(Solver):
    """Solver for NLP problems that uses Ipopt."""
    name = 'ipopt'

    description = 'NLP solver.'

    def actual_solve(self, problem, **kwargs):
        if len(problem.objectives) != 1:
            raise ValueError('Problem must have exactly 1 objective function.')
        logger = Logger.from_kwargs(**kwargs)
        xi, xl, xu = self.get_starting_point_and_bounds(problem)
        gl, gu = self.get_constraints_bounds(problem)
        ipopt_solution = ipopt_solve(problem, xi, xl, xu, gl, gu)
        return self._build_solution(problem, ipopt_solution)

    def _build_solution(self, problem, solution):
        status = IpoptStatus(solution.status)
        opt_obj = OptimalObjective(
            name=problem.objectives[0].name,
            value=solution.objective_value)
        opt_vars = [
            OptimalVariable(name=variable.name, value=solution.x[i])
            for i, variable in enumerate(problem.variables)
        ]
        return Solution(status, optimal_obj=opt_obj, optimal_vars=opt_vars)

    def _build_optimal_variable(self, i, variable, solution):
        OptimalVariable(name=variable.name, value=solution.x[i])

    def get_starting_point_and_bounds(self, problem):
        nx = problem.num_variables
        xi = np.zeros(nx)
        xl = np.zeros(nx)
        xu = np.zeros(nx)
        for i in range(nx):
            v = problem.variable_view(i)
            if v.has_starting_point():
                x[i] = v.starting_point()
            else:
                lb = v.lower_bound()
                lb = lb if lb is not None else -2e19

                ub = v.upper_bound()
                ub = ub if ub is not None else 2e19

                xi[i] = max(lb, min(ub, 0))
                xl[i] = lb
                xu[i] = ub
        return xi, xl, xu

    def get_constraints_bounds(self, problem):
        ng = problem.num_constraints
        gl = np.zeros(ng)
        gu = np.zeros(ng)
        for i in range(ng):
            c = problem.constraint(i)
            gl[i] = c.lower_bound if c.lower_bound is not None else -2e19
            gu[i] = c.upper_bound if c.upper_bound is not None else 2e19
        return gl, gu
