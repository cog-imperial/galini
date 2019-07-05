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

"""Ipopt solution and status."""
import numpy as np
from galini.logging import get_logger
from galini.math import mc, almost_ge, almost_le
from galini.core import (
    IpoptSolution as CoreIpoptSolution,
)
from galini.solvers import (
    Solution,
    Status,
    OptimalObjective,
    OptimalVariable,
)


logger = get_logger(__name__)


def build_solution(run_id, problem, solution, tree_data, out_indexes):
    if not solution.x:
        status = IpoptStatusInfeasible(solution.status)
        opt_obj = None
        opt_vars = None
    else:

        fg = tree_data.eval(solution.x, out_indexes)
        fg_x = fg.forward(0, solution.x)

        obj_value = fg_x[0]

        opt_obj = OptimalObjective(
            name=problem.objectives[0].name,
            value=obj_value,
        )

        opt_vars = [
            OptimalVariable(name=variable.name, value=solution.x[i])
            for i, variable in enumerate(problem.variables)
        ]

        if _solution_is_feasible(run_id, problem, solution, fg_x):
            status = IpoptStatusSuccess(solution.status)
        else:
            status = IpoptStatusInfeasible(solution.status)

    return IpoptSolution(
        status,
        optimal_obj=opt_obj,
        optimal_vars=opt_vars,
        zl=np.array(solution.zl),
        zu=np.array(solution.zu),
        g=np.array(solution.g),
        lambda_=np.array(solution.lambda_)
    )


def _solution_is_feasible(run_id, problem, solution, fg_x):
    logger.debug(run_id, 'Checking infeasibility')
    i = 0
    for _, constraint in enumerate(problem.constraints):
        if constraint.metadata.get('rlt_constraint_info'):
            continue
        lb = constraint.lower_bound
        if lb is None:
            lb = -np.inf
        ub = constraint.upper_bound
        if ub is None:
            ub = np.inf
        logger.debug(run_id, 'Con {}: {} <= {} <= {}', constraint.name, lb, fg_x[i+1], ub)
        if not almost_le(lb, fg_x[i+1], atol=mc.constraint_violation_tol):
            i += 1
            return False
        if not almost_le(fg_x[i+1], ub, atol=mc.constraint_violation_tol):
            i += 1
            return False
        i += 1
    return True


class IpoptStatusSuccess(Status):
    def __init__(self, inner):
        self.inner = inner

    def is_success(self):
        return True

    def is_infeasible(self):
        return False

    def is_unbounded(self):
        return False

    def description(self):
        return 'Success'


class IpoptStatusInfeasible(Status):
    def __init__(self, inner):
        self.inner = inner

    def is_success(self):
        return False

    def is_infeasible(self):
        return True

    def is_unbounded(self):
        return False

    def description(self):
        return 'Infeasible'


class IpoptSolution(Solution):
    def __init__(self, status, optimal_obj=None, optimal_vars=None,
                 zl=None, zu=None, g=None, lambda_=None):
        super().__init__(status, optimal_obj, optimal_vars)
        self.zl = zl
        self.zu = zu
        self.g = g
        self.lambda_ = lambda_

    def __str__(self):
        return 'IpoptSolution(status={}, objective={})'.format(
            self.status.description(),
            self.objectives,
        )

    def __repr__(self):
        return '<{} at {}>'.format(str(self), hex(id(self)))
