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
import pyomo.environ as pe

from galini.pyomo import safe_setub, safe_setlb
from galini.math import is_inf
from galini.solvers.solution import load_solution_from_model


def solve_primal(model, mip_solution, solver, mc):
    """Solve primal by fixing integer variables and solving the NLP.

    If the search fails and f `mip_solution` has a solution pool, then also
    try to find a feasible solution starting at the solution pool points.

    Parameters
    ----------
    model : ConcreteModel
        the mixed integer, (possibly) non convex problem
    mip_solution : MipSolution
        the linear relaxation solution
    solver : Solver
        the NLP solver used to solve the problem
    """
    # starting_point = [v.value for v in mip_solution.variables]
    solution = solve_primal_with_starting_point(
        model, mip_solution, solver, mc
    )

    if solution.status.is_success():
        return solution
    # TODO(fra): update to use solution pool again
    return solution


def solve_primal_with_starting_point(model, starting_point, solver, mc, fix_all=False):
    """Solve primal using mip_solution as starting point and fixing variables.

    Parameters
    ----------
    model : ConcreteModel
        the mixed integer, (possibly) non convex problem
    starting_point : dict-like
        the starting point for each variable
    solver : Solver
        the NLP solver used to solve the problem
    fix_all
        if `True`, fix all variables, otherwise fix integer variables only

    Returns
    -------
    A solution to the problem
    """
    assert isinstance(starting_point, pe.ComponentMap)

    fixed_vars = []
    for var in model.component_data_objects(pe.Var, active=True):
        # If the user did not specify a value for the variable reset it
        if var.is_fixed():
            continue
        user_value = starting_point.get(var, None)
        if not var.is_continuous() and user_value is not None:
            user_value = int(user_value)
        var.set_value(user_value)
        point = compute_variable_starting_point(var, mc)
        var.set_value(point)

        # Fix integers variables
        # We set bounds to avoid issues with ipopt
        lb = var.lb
        ub = var.ub
        if not var.is_continuous() or fix_all:
            safe_setlb(var, point)
            safe_setub(var, point)
            fixed_vars.append((var, lb, ub))

    try:
        results = solver.solve(model, tee=True)

        # unfix all variables
        for var, lb, ub in fixed_vars:
            safe_setlb(var, lb)
            safe_setub(var, ub)
    except Exception as ex:
        # unfix all variables, then rethrow
        for var, lb, ub in fixed_vars:
            safe_setlb(var, lb)
            safe_setub(var, ub)
        raise ex

    return load_solution_from_model(results, model)


def compute_variable_starting_point(var, mc):
    """Compute a variable starting point, using its value if present."""
    point = _compute_variable_starting_point_as_float(var, mc)
    if var.is_continuous():
        return point
    return int(point)


def _compute_variable_starting_point_as_float(var, mc):
    # Use starting point if present
    value = pe.value(var, exception=False)
    if value is not None:
        return value
    # If var has both bounds, use midpoint
    lb = var.lb
    if is_inf(lb, mc):
        lb = None
    ub = var.ub
    if is_inf(ub, mc):
        ub = None
    if lb is not None and ub is not None:
        return lb + 0.5 * (ub - lb)
    # If unbounded, use 0
    if lb is None and lb is None:
        return 0.0
    # If no lower bound, use upper bound
    if lb is None:
        return ub
    # Otherwise, use lower bound
    return lb
