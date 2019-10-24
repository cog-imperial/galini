#  Copyright 2019 Francesco Ceccon
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Functions to solve primal problem."""
import numpy as np

from galini.core import Domain
from galini.math import is_close, mc
from galini.timelimit import seconds_left


def solve_primal(run_id, problem, mip_solution, solver):
    """Solve primal by fixing integer variables and solving the NLP.

    If the search fails and f `mip_solution` has a solution pool, then also
    try to find a feasible solution starting at the solution pool points.

    Parameters
    ----------
    run_id : str
        the run_id used for logging
    problem : Problem
        the mixed integer, (possibly) non convex problem
    mip_solution : MipSolution
        the linear relaxation solution
    solver : Solver
        the NLP solver used to solve the problem
    """
    solution = solve_primal_with_solution(run_id, problem, mip_solution, solver)
    if solution.status.is_success():
        return solution
    # Try solutions from mip solution pool, if available
    if mip_solution.solution_pool is None:
        return solution
    for mip_solution_from_pool in mip_solution.solution_pool:
        if seconds_left() <= 0:
            return solution
        solution_from_pool = solve_primal_with_solution(
            run_id, problem, mip_solution_from_pool.inner, solver
        )
        if solution_from_pool.status.is_success():
            return solution_from_pool
    # No solution from pool was feasible, return original infeasible sol
    return solution



def solve_primal_with_solution(
        run_id, problem, mip_solution, solver, fix_all=False):
    """Solve primal using mip_solution as starting point and fixing variables.

    Parameters
    ----------
    run_id
        the run_id used for logging
    problem
        the mixed integer, (possibly) non convex problem
    mip_solution
        the linear relaxation solution
    solver : Solver
        the NLP solver used to solve the problem
    fix_all
        if `True`, fix all variables, otherwise fix integer variables only
    Returns
    -------
    A solution to the problem
    """
    for v, sv in zip(problem.variables, mip_solution.variables):
        domain = problem.domain(v)
        view = problem.variable_view(v)
        if sv.value is None:
            lb = view.lower_bound()
            if lb is None:
                lb = -mc.infinity
            ub = view.upper_bound()
            if ub is None:
                ub = mc.infinity

            value = lb + (ub - lb) / 2.0
        else:
            value = sv.value

        if domain != Domain.REAL:
            # Solution (from pool) can contain non integer values for
            # integer variables. Simply round these values up
            if not is_close(np.trunc(value), value, atol=mc.epsilon):
                value = min(view.upper_bound(), np.ceil(value))
            problem.fix(v, value)
        elif fix_all:
            problem.fix(v, value)
        else:
            problem.set_starting_point(v, value)

    try:
        solution = solver.solve(problem)

        # unfix all variables
        for v in problem.variables:
            problem.unfix(v)
    except ex:
        # unfix all variables, then rethrow
        for v in problem.variables:
            problem.unfix(v)
        raise ex

    return solution
