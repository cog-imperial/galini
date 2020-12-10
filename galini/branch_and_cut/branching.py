#  Copyright 2020 Francesco Ceccon
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

"""Compute branching decision after visiting a node."""

import numpy as np
import pyomo.environ as pe
from coramin.relaxations.mccormick import PWMcCormickRelaxation

from galini.branch_and_bound.branching import BranchingPoint
from galini.branch_and_bound.strategy import BranchingStrategy
from galini.branch_and_bound.strategy import least_reduced_variable
from galini.branch_and_cut.node_storage import BranchingDecision
from galini.math import is_close, is_inf


BILINEAR_RELAXATIONS_TYPES = (PWMcCormickRelaxation,)


class BranchAndCutBranchingStrategy(BranchingStrategy):
    def __init__(self, algorithm):
        pass

    def branch(self, node, tree):
        branching_decision = node.storage.branching_decision
        if branching_decision is None:
            return None
        assert branching_decision.variable is not None
        assert branching_decision.variable is not None
        return BranchingPoint(
            branching_decision.variable, branching_decision.point
        )


def compute_branching_decision(model, linear_model, root_bounds, mip_solution, weights, lambda_, mc):
    """Returns branching variable and point, or None if no branching exists.

    Parameters
    ----------
    model : ConcreteModel
        the user model
    linear_model : ConcreteModel
        the linear relaxation of the user model
    root_bounds : dict-like
        variables bounds at the root node
    mip_solution : Solution
        solution to linear_problem
    weights : dict
        weights for nonlinear infeasibility
    lambda_ : float
        weight for convex combination of midpoint and solution

    Returns
    -------
    BranchingDecision
        branching variable and point, or None if no branching should be
        performed
    """
    # Variables in [-inf, inf] have priority.
    unbounded_variable = _unbounded_variable(model)
    if unbounded_variable is not None:
        return BranchingDecision(variable=unbounded_variable, point=0.0)

    branching_variable = compute_branching_variable(
        model, linear_model, mip_solution, weights, mc
    )
    if branching_variable is None:
        branching_variable = least_reduced_variable(model, root_bounds)
        if branching_variable is None:
            return None
    linear_branching_variable = \
        linear_model.find_component(branching_variable.getname(fully_qualified=True))
    point = compute_branching_point(linear_branching_variable, mip_solution, lambda_, mc)
    return BranchingDecision(variable=branching_variable, point=point)


def compute_branching_variable(problem, linear_problem, mip_solution,
                               weights, mc):
    """Branch on variable with max nonlinear infeasibility.

    Parameters
    ----------
    problem : Problem
        the user problem
    linear_problem : Problem
        the linear relaxation of problem
    mip_solution : Solution
        the solution to linear_problem
    weights : dict
        the weights for the sum, max, and min nonlinear components

    Returns
    -------
    Variable
        the branching variable or None if it should not branch

    """
    nonlinear_infeasibility = \
        compute_nonlinear_infeasiblity_components(linear_problem, mip_solution)

    # score each variable and pick the one with the maximum
    # ignore variables that can be considered "fixed"
    branching_var = None
    branching_var_score = None
    for var in nonlinear_infeasibility['sum'].keys():
        lb, ub = var.bounds
        if not is_inf(lb, mc) and not is_inf(ub, mc):
            if is_close(lb, ub, rtol=mc.epsilon):
                continue

        if branching_var is None:
            branching_var = var
            branching_var_score = _infeasibility_score(
                var, nonlinear_infeasibility, weights
            )
        else:
            var_score = _infeasibility_score(
                var, nonlinear_infeasibility, weights
            )
            if var_score > branching_var_score:
                branching_var = var
                branching_var_score = var_score

    if branching_var is None:
        return None

    return problem.find_component(branching_var.getname(fully_qualified=True))


def compute_nonlinear_infeasiblity_components(linear_problem, mip_solution):
    """Compute the sum, min, and max of the nonlinear infeasibility.

    Parameters
    ----------
    linear_problem : Problem
        the linear relaxation of the problem
    mip_solution : Solution
        solution to linear_problem

    References
    ----------
    Belotti, P., Lee, J., Liberti, L., Margot, F., & Wächter, A. (2009).
        Branching and bounds tightening techniques for non-convex MINLP.
        Optimization Methods and Software, 24(4–5), 597–634.
    """
    nonlinear_infeasibility_sum = pe.ComponentMap()
    nonlinear_infeasibility_min = pe.ComponentMap()
    nonlinear_infeasibility_max = pe.ComponentMap()

    if not mip_solution.status.is_success():
        return {
            'sum': nonlinear_infeasibility_sum,
            'max': nonlinear_infeasibility_max,
            'min': nonlinear_infeasibility_min,
        }

    for relaxation in linear_problem.galini_nonlinear_relaxations:
        if not isinstance(relaxation, BILINEAR_RELAXATIONS_TYPES):
            continue
        rhs_vars = relaxation.get_rhs_vars()

        if len(rhs_vars) > 2:
            continue

        aux_var = relaxation.get_aux_var()

        if len(rhs_vars) == 2:
            v1, v2 = rhs_vars
        else:
            assert len(rhs_vars) == 1
            v1 = v2 = rhs_vars[0]
        v1_xk = mip_solution.variables[v1]
        v2_xk = mip_solution.variables[v2]
        aux_xk = mip_solution.variables[aux_var]

        # U(x_k) = |x_ik - v_i(x_k)| / (1 + ||grad(v_i(x_k))||)
        bilinear_discrepancy = np.abs(aux_xk - v1_xk*v2_xk)
        scaling = (1 + np.sqrt(v1_xk**2.0 + v2_xk**2.0))
        err = bilinear_discrepancy / scaling

        # First time we see v1
        if v1 not in nonlinear_infeasibility_sum:
            nonlinear_infeasibility_sum[v1] = 0.0
            nonlinear_infeasibility_max[v1] = -np.inf
            nonlinear_infeasibility_min[v1] = np.inf

        # First time we see v2
        if v2 not in nonlinear_infeasibility_sum:
            nonlinear_infeasibility_sum[v2] = 0.0
            nonlinear_infeasibility_max[v2] = -np.inf
            nonlinear_infeasibility_min[v2] = np.inf

        nonlinear_infeasibility_sum[v1] += err
        nonlinear_infeasibility_sum[v2] += err

        nonlinear_infeasibility_max[v1] = max(
            nonlinear_infeasibility_max[v1],
            err,
        )
        nonlinear_infeasibility_max[v2] = max(
            nonlinear_infeasibility_max[v2],
            err,
        )

        nonlinear_infeasibility_min[v1] = min(
            nonlinear_infeasibility_min[v1],
            err,
        )
        nonlinear_infeasibility_min[v2] = min(
            nonlinear_infeasibility_min[v2],
            err,
        )

    return {
        'sum': nonlinear_infeasibility_sum,
        'max': nonlinear_infeasibility_max,
        'min': nonlinear_infeasibility_min,
    }


def compute_branching_point(var, mip_solution, lambda_, mc):
    """Compute a convex combination of the midpoint and the solution value.

    Given a variable $x_i \in [x_i^L, x_i^U]$, with midpoint
    $x_m = x_i^L + 0.5(x_i^U - x_i^L)$, and its solution value $\bar{x_i}$,
    branch at $\lambda x_m + (1 - \lambda) \bar{x_i}$.

    Parameters
    ----------
    var : Var
        the branching variable
    mip_solution : Solution
        the MILP solution
    lambda_ : float
        the weight
    """
    x_bar = mip_solution.variables[var]
    lb = var.lb
    ub = var.ub

    # Use special bounds if variable is unbounded
    user_upper_bound = mc.user_upper_bound
    if not var.is_continuous():
        user_upper_bound = mc.user_integer_upper_bound

    has_lb = True
    if lb is None or is_inf(lb, mc):
        has_lb = False
        lb = -user_upper_bound

    has_ub = True
    if ub is None or is_inf(ub, mc):
        has_ub = False
        ub = user_upper_bound

    # if unbounded on either side (or both), branch at midpoint
    if x_bar is not None and (not has_lb or not has_ub):
        return x_bar

    midpoint = lb + 0.5 * (ub - lb)
    if x_bar is None:
        return midpoint
    # The solver may have returned a point outside of the bounds.
    # Avoid propagating the error
    if x_bar < lb or x_bar > ub:
        return midpoint
    return lambda_ * midpoint + (1.0 - lambda_) * x_bar


def _unbounded_variable(model):
    for var in model.component_data_objects(pe.Var, active=True):
        unbounded = not var.has_lb() and not var.has_ub()
        if unbounded:
            return var
    return None


def _infeasibility_score(var, nonlinear_infeasibility, weights):
    return (
            nonlinear_infeasibility['sum'][var] * weights['sum'] +
            nonlinear_infeasibility['max'][var] * weights['max'] +
            nonlinear_infeasibility['min'][var] * weights['min']
    )
