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

from galini.branch_and_cut.node_storage import BranchingDecision
from galini.math import mc, is_close, is_inf
from galini.branch_and_bound.strategy import BranchingStrategy
from galini.branch_and_bound.branching import BranchingPoint


class BranchAndCutBranchingStrategy(BranchingStrategy):
    def branch(self, node, tree):
        branching_decision = node.storage.branching_decision
        if branching_decision is None:
            return None
        assert branching_decision.variable is not None
        assert branching_decision.variable is not None
        return BranchingPoint(
            branching_decision.variable, branching_decision.point
        )


def compute_branching_decision(problem, linear_problem, mip_solution, weights,
                               lambda_):
    """Returns branching variable and point, or None if no branching exists.

    Parameters
    ----------
    problem : Problem
        the user problem
    linear_problem : Problem
        the linear relaxation of problem
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
    unbounded_variable = _unbounded_variable(problem)
    if unbounded_variable is not None:
        return BranchingDecision(variable=unbounded_variable, point=0.0)

    branching_variable = compute_branching_variable(
        problem, linear_problem, mip_solution, weights
    )
    if branching_variable is None:
        return None
    point = compute_branching_point(branching_variable, mip_solution, lambda_)
    return BranchingDecision(variable=branching_variable, point=point)


def compute_branching_variable(problem, linear_problem, mip_solution, weights):
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
    for var_idx in nonlinear_infeasibility['sum'].keys():
        vv = problem.variable_view(var_idx)

        if is_close(vv.lower_bound(), vv.upper_bound(), rtol=mc.epsilon):
            continue

        if branching_var is None:
            branching_var = var_idx
            branching_var_score = _infeasibility_score(
                var_idx, nonlinear_infeasibility, weights
            )
        else:
            var_score = _infeasibility_score(
                var_idx, nonlinear_infeasibility, weights
            )
            if var_score > branching_var_score:
                branching_var = var_idx
                branching_var_score = var_score

    if branching_var is None:
        return None
    return problem.variable_view(branching_var)


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
        Branching and bounds tighteningtechniques for non-convex MINLP.
        Optimization Methods and Software, 24(4–5), 597–634.
    """
    nonlinear_infeasibility_sum = dict()
    nonlinear_infeasibility_min = dict()
    nonlinear_infeasibility_max = dict()

    for var in linear_problem.variables:
        if not mip_solution.status.is_success():
            continue

        if var.reference:
            if not hasattr(var.reference, 'var1'):
                continue
            v1 = var.reference.var1
            v2 = var.reference.var2

            w_xk = mip_solution.variables[var.idx].value
            v1_xk = mip_solution.variables[v1.idx].value
            v2_xk = mip_solution.variables[v2.idx].value

            if v1_xk is None or v2_xk is None:
                continue


            # U(x_k) = |x_ik - v_i(x_k)| / (1 + ||grad(v_i(x_k))||)
            bilinear_discrepancy = np.abs(w_xk - v1_xk*v2_xk)
            scaling = (1 + np.sqrt(v1_xk**2.0 + v2_xk**2.0))
            err = bilinear_discrepancy / scaling

            # First time we see v1
            if v1.idx not in nonlinear_infeasibility_sum:
                nonlinear_infeasibility_sum[v1.idx] = 0.0
                nonlinear_infeasibility_max[v1.idx] = -np.inf
                nonlinear_infeasibility_min[v1.idx] = np.inf

            # First time we see v2
            if v2.idx not in nonlinear_infeasibility_sum:
                nonlinear_infeasibility_sum[v2.idx] = 0.0
                nonlinear_infeasibility_max[v2.idx] = -np.inf
                nonlinear_infeasibility_min[v2.idx] = np.inf

            nonlinear_infeasibility_sum[v1.idx] += err
            nonlinear_infeasibility_sum[v2.idx] += err

            nonlinear_infeasibility_max[v1.idx] = max(
                nonlinear_infeasibility_max[v1.idx],
                err,
            )
            nonlinear_infeasibility_max[v2.idx] = max(
                nonlinear_infeasibility_max[v2.idx],
                err,
            )

            nonlinear_infeasibility_min[v1.idx] = min(
                nonlinear_infeasibility_min[v1.idx],
                err,
            )
            nonlinear_infeasibility_min[v2.idx] = min(
                nonlinear_infeasibility_min[v2.idx],
                err,
            )

    return {
        'sum': nonlinear_infeasibility_sum,
        'max': nonlinear_infeasibility_max,
        'min': nonlinear_infeasibility_min,
    }


def compute_branching_point(variable, mip_solution, lambda_):
    """Compute a convex combination of the midpoint and the solution value.

    Given a variable $x_i \in [x_i^L, x_i^U]$, with midpoint
    $x_m = x_i^L + 0.5(x_i^U - x_i^L)$, and its solution value $\bar{x_i}$,
    branch at $\lambda x_m + (1 - \lambda) \bar{x_i}$.

    Parameters
    ----------
    variable : VariableView
        the branching variable
    mip_solution : Solution
        the MILP solution
    lambda_ : float
        the weight
    """
    x_bar = mip_solution.variables[variable.variable.idx].value
    lb = variable.lower_bound()
    ub = variable.upper_bound()

    # Use special bounds if variable is unbounded
    user_upper_bound = mc.user_upper_bound
    if variable.domain.is_integer():
        user_upper_bound = mc.user_integer_upper_bound

    if is_inf(lb):
        lb = -user_upper_bound

    if is_inf(ub):
        ub = user_upper_bound

    midpoint = lb + 0.5 * (ub - lb)
    return lambda_ * midpoint + (1.0 - lambda_) * x_bar



def _unbounded_variable(problem):
    for var in problem.variables:
        unbounded = (
           is_inf(problem.lower_bound(var)) and
           is_inf(problem.upper_bound(var))
        )
        if unbounded:
            return var
    return None


def _infeasibility_score(var_idx, nonlinear_infeasibility, weights):
    return (
        nonlinear_infeasibility['sum'][var_idx] * weights['sum'] +
        nonlinear_infeasibility['max'][var_idx] * weights['max'] +
        nonlinear_infeasibility['min'][var_idx] * weights['min']
    )
