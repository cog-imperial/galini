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
from galini.math import is_inf
from galini.pyomo import safe_setub, safe_setlb
from galini.solvers.solution import load_solution_from_model
from galini.branch_and_bound.node import NodeSolution
from pyutilib.common import ApplicationError


class InitialPrimalSearchStrategy:
    """Initial primal search strategy interface."""
    def solve(self, model, tree, node):
        raise NotImplementedError('InitialPrimalSearchStrategy.solve')


class PrimalHeuristic:
    def solve(self, model, linear_model, mip_solution, tree, node):
        raise NotImplementedError('PrimalHeuristic.solve')


class NoInitialPrimalSearchStrategy(InitialPrimalSearchStrategy):
    """An initial primal search strategy that does nothing."""
    def __init__(self, algorithm):
        pass

    def solve(self, model, tree, node):
        return None


class DefaultInitialPrimalSearchStrategy(InitialPrimalSearchStrategy):
    """An initial primal search strategy that fixes integer variables and solves the NLP."""
    def __init__(self, algorithm):
        self.algorithm = algorithm

    def solve(self, model, tree, node):
        _, _, cvx = self.algorithm._perform_fbbt_on_model(tree, node, model, maxiter=1)

        solution = self.algorithm._try_solve_convex_model(model, convexity=cvx)
        if solution is not None:
            solution.best_obj_estimate = solution.objective
            return NodeSolution(solution, solution)

        # Don't pass a starting point since it's already loaded in the model
        timelimit = min(
            self.algorithm.bab_config['root_node_feasible_solution_search_timelimit'],
            self.algorithm.galini.timelimit.seconds_left(),
        )
        self.algorithm._update_solver_options(self.algorithm._nlp_solver, timelimit=timelimit)
        solution = solve_primal_with_starting_point(
            model, pe.ComponentMap(), self.algorithm._nlp_solver, self.algorithm.galini.mc
        )

        if solution is None:
            return None

        if solution.status.is_success():
            return NodeSolution(None, solution)
        return None


class DefaultPrimalHeuristic(PrimalHeuristic):
    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.galini = algorithm.galini

    def solve(self, model, linear_model, mip_solution, tree, node):
        assert mip_solution.status.is_success(), "Should be a feasible point for the relaxation"

        model_var_map = node.storage.model_to_relaxation_var_map
        mip_solution_with_model_vars = pe.ComponentMap(
            (var, mip_solution.variables[model_var_map[var]])
            for var in model.component_data_objects(pe.Var, active=True)
        )

        self.algorithm._update_solver_options(self.algorithm._nlp_solver)
        primal_solution = solve_primal_with_starting_point(
            model, mip_solution_with_model_vars, self.algorithm._nlp_solver, self.galini.mc, fix_all=True
        )

        if primal_solution is not None and primal_solution.status.is_success():
            return primal_solution

        self.algorithm._update_solver_options(self.algorithm._nlp_solver)
        new_primal_solution = solve_primal(
            model, mip_solution_with_model_vars, self.algorithm._nlp_solver, self.galini.mc
        )

        if new_primal_solution is not None:
            primal_solution = new_primal_solution

        return primal_solution


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
    solution = solve_primal_with_starting_point(
        model, mip_solution, solver, mc
    )

    if solution is None:
        return None

    if solution.status.is_success():
        return solution

    mip_solution_pool = getattr(mip_solution, 'solution_pool', None)
    if mip_solution_pool is not None:
        for mip_solution in mip_solution_pool:
            solution_from_pool = solve_primal_with_starting_point(
                model, mip_solution, solver, mc
            )
            if solution_from_pool.status.is_success():
                return solution_from_pool

    # No solution was feasible, return original infeasible solution.
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
    mc : MathContext
        the math context used for computations
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
        else:
            point = compute_variable_starting_point(var, mc, user_value)
            var.set_value(point)

        # Fix integers variables
        # We set bounds to avoid issues with ipopt
        lb = var.lb
        ub = var.ub
        if not var.is_continuous() or fix_all:
            point = pe.value(var)
            safe_setlb(var, point)
            safe_setub(var, point)
            fixed_vars.append((var, lb, ub))

    try:
        results = solver.solve(model, tee=False)

        # unfix all variables
        for var, lb, ub in fixed_vars:
            safe_setlb(var, lb)
            safe_setub(var, ub)
    except (ValueError, ApplicationError):
        for var, lb, ub in fixed_vars:
            safe_setlb(var, lb)
            safe_setub(var, ub)
        return None
    except Exception as ex:
        # unfix all variables, then rethrow
        for var, lb, ub in fixed_vars:
            safe_setlb(var, lb)
            safe_setub(var, ub)
        raise

    return load_solution_from_model(results, model, solver=solver)


def compute_variable_starting_point(var, mc, user_value):
    """Compute a variable starting point, using its value if present."""
    point = _compute_variable_starting_point_as_float(var, mc, user_value)
    if var.is_continuous():
        return point
    return int(point)


def _compute_variable_starting_point_as_float(var, mc, value):
    # Use starting point if present
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
