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
"""MIP solver class."""
import pulp

from galini.config import SolverOptions, ExternalSolverOptions
from galini.core import Problem
from galini.logging import get_logger
from galini.mip.pulp import dag_to_pulp, mip_solver
from galini.mip.solution import MIPSolution, PulpStatus
from galini.solvers import Solver, OptimalVariable, OptimalObjective
from galini.solvers.solution import SolutionPool

logger = get_logger(__name__)


class MIPSolver(Solver):
    """Solver class that uses COIN-OR pulp to solve linear problems."""
    name = 'mip'

    description = 'MIP Solver that delegates to Cplex or CBC.'

    @staticmethod
    def solver_options():
        """Return options for this solver."""
        return SolverOptions(MIPSolver.name, [
            ExternalSolverOptions('cplex'),
        ])

    def actual_solve(self, problem, run_id, **kwargs):
        solver = mip_solver(logger, run_id, self.galini)
        if isinstance(problem, Problem):
            # Convert to pulp and then solve it.
            assert problem.objective
            pulp_problem, variables = dag_to_pulp(problem)
            status = pulp_problem.solve(solver)

            optimal_variables = [
                OptimalVariable(var.name, pulp.value(variables[var.idx]))
                for var in problem.variables
            ]

            dual_values = [cons.pi for cons in pulp_problem.constraints.values()]
            if all(dv is None for dv in dual_values):
                dual_values = None

            if getattr(solver, 'solution_pool'):
                pool = SolutionPool(10)
                for (obj, x_i) in solver.solution_pool:
                    pool_vars = [
                        OptimalVariable(var.name, x_i.get(var.name, None))
                        for var in problem.variables
                    ]

                    pool.add(
                        MIPSolution(
                            PulpStatus(pulp.LpStatusOptimal), # Not optimal, but feasible
                            OptimalObjective(problem.objective.name, obj),
                            pool_vars,
                        )
                    )
            else:
                pool = None

            return MIPSolution(
                PulpStatus(status),
                OptimalObjective(problem.objective.name, pulp.value(pulp_problem.objective)),
                optimal_variables,
                dual_values,
                pool=pool,
            )

        # Problem is a pulp model. Solve this instead.
        status = problem.solve(solver)

        dual_values = [cons.pi for cons in problem.constraints.values()]
        if all(dv is None for dv in dual_values):
            dual_values = None

        return MIPSolution(
            PulpStatus(status),
            OptimalObjective(problem.objective.name, pulp.value(problem.objective)),
            [OptimalVariable(var.name, pulp.value(var)) for var in problem.variables()],
            dual_values,
        )
