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

"""OBBT step of branch-and-cut algorithm."""
import coramin.domain_reduction.obbt as coramin_obbt
import coramin.domain_reduction.filters as coramin_filters
import numpy as np
import pyomo.environ as pe
from coramin.relaxations.iterators import relaxation_data_objects
from galini.branch_and_cut.branching import BILINEAR_RELAXATIONS_TYPES
from galini.math import is_close
from galini.pyomo import safe_set_bounds
from galini.pyomo.util import update_solver_options
from galini.timelimit import (
    current_time,
    seconds_elapsed_since,
)
from pyomo.core.kernel.component_set import ComponentSet
from suspect.fbbt import perform_fbbt
from suspect.interval import Interval
from suspect.propagation import propagate_special_structure

coramin_logger = coramin_obbt.logger  # pylint: disable=invalid-name
coramin_logger.disabled = True


def perform_obbt_on_model(solver, model, linear_model, upper_bound, timelimit, relative_gap, absolute_gap,
                          simplex_maxiter, mc):
    """Perform OBBT on Pyomo model using Coramin.

    Parameters
    ----------
    solver : Solver
        the mip solver to use
    model : ConcreteModel
        the pyomo concrete model
    linear_model : ConcreteModel
        the linear relaxation of model
    upper_bound : float or None
        the objective value upper bound, if known
    timelimit : int
        a timelimit, in seconds
    relative_gap : float
        mip relative gap
    absolute_gap : float
        mip absolute gap
    simplex_maxiter : int
        the maximum number of simplex iterations
    mc : MathContext
        GALINI math context
    """
    obbt_start_time = current_time()

    originally_integer = []
    original_bounds = pe.ComponentMap()
    for var in linear_model.component_data_objects(ctype=pe.Var):
        original_bounds[var] = var.bounds
        if var.is_continuous():
            originally_integer.append((var, var.domain))
            var.domain = pe.Reals

    # collect variables in nonlinear constraints
    nonlinear_variables = ComponentSet()
    for relaxation in relaxation_data_objects(linear_model, active=True, descend_into=True):
        if not isinstance(relaxation, BILINEAR_RELAXATIONS_TYPES):
            for var in relaxation.get_rhs_vars():
                # Coramin will complain about variables that are fixed
                if not var.has_lb() or not var.has_ub():
                    nonlinear_variables.add(var)
                else:
                    if not np.abs(var.ub - var.lb) < mc.epsilon:
                        nonlinear_variables.add(var)

    time_left = timelimit - seconds_elapsed_since(obbt_start_time)
    nonlinear_variables = list(nonlinear_variables)

    update_solver_options(
        solver,
        timelimit=time_left,
        maxiter=simplex_maxiter,
        relative_gap=relative_gap,
        absolute_gap=absolute_gap,
    )

    obbt_ex = None
    result = None
    try:
        (vars_to_minimize, vars_to_maximize) = \
            coramin_filters.aggressive_filter(
                candidate_variables=nonlinear_variables,
                relaxation=linear_model,
                solver=solver,
                objective_bound=upper_bound,
                tolerance=mc.epsilon,
                max_iter=10,
                improvement_threshold=5
            )
        vars_to_tighten = vars_to_minimize
        visited_vars = ComponentSet(vars_to_tighten)
        for v in vars_to_maximize:
            if v not in visited_vars:
                vars_to_tighten.append(v)
                visited_vars.add(v)
        result = coramin_obbt.perform_obbt(
            linear_model,
            solver,
            time_limit=time_left,
            varlist=vars_to_tighten,
            objective_bound=upper_bound,
            warning_threshold=mc.epsilon
        )
    except Exception as ex:
        obbt_ex = ex

    for var, domain in originally_integer:
        var.domain = domain

    # If we encountered an exception in Coramin, restore bounds and then raise.
    if obbt_ex is not None:
        for var, (lb, ub) in original_bounds.items():
            var.setlb(lb)
            var.setub(ub)
        raise obbt_ex

    if result is None:
        return

    new_bounds = pe.ComponentMap()

    eps = mc.epsilon

    for var, new_lb, new_ub in zip(nonlinear_variables, *result):
        original_var = model.find_component(var.getname(fully_qualified=True))
        if original_var is None:
            continue
        new_lb = best_lower_bound(var, new_lb, var.lb, eps)
        new_ub = best_upper_bound(var, new_ub, var.ub, eps)
        if np.abs(new_ub - new_lb) < eps:
            new_lb = new_ub
        new_bounds[var] = (new_lb, new_ub)
        safe_set_bounds(var, new_lb, new_ub)
        safe_set_bounds(original_var, new_lb, new_ub)

    # Rebuild relaxations since bounds changed
    for relaxation in relaxation_data_objects(linear_model, active=True, descend_into=True):
        relaxation.rebuild()

    return new_bounds


def perform_fbbt_on_model(model, tree, node, maxiter, timelimit, eps, skip_special_structure=False):
    """

    Parameters
    ----------
    model
    tree
    node
    maxiter
    timelimit
    eps

    Returns
    -------

    """
    objective_bounds = pe.ComponentMap()
    objective_bounds[model._objective] = (tree.lower_bound, tree.upper_bound)

    branching_variable = None
    if not node.storage.is_root:
        branching_variable = node.storage.branching_variable

    fbbt_start_time = current_time()
    should_continue = lambda: seconds_elapsed_since(fbbt_start_time) <= timelimit

    bounds = perform_fbbt(
        model,
        max_iter=maxiter,
        objective_bounds=objective_bounds,
        should_continue=should_continue,
        #branching_variable=branching_variable,
    )

    if not skip_special_structure:
        monotonicity, convexity = \
            propagate_special_structure(model, bounds)
    else:
        monotonicity = convexity = None

    cause_infeasibility = None
    new_bounds_map = pe.ComponentMap()
    for var in model.component_data_objects(pe.Var, active=True):
        new_bound = bounds[var]
        if new_bound is None:
            new_bound = Interval(None, None)

        new_lb = best_lower_bound(var, new_bound.lower_bound, var.lb, eps)
        new_ub = best_upper_bound(var, new_bound.upper_bound, var.ub, eps)

        new_bounds_map[var] = (new_lb, new_ub)
        if new_lb > new_ub:
            cause_infeasibility = var
            break

    if cause_infeasibility is not None:
        return None, None, None
    else:
        for var, (new_lb, new_ub) in new_bounds_map.items():
            if np.abs(new_ub - new_lb) < eps:
                new_lb = new_ub
            safe_set_bounds(var, new_lb, new_ub)
            # Also update bounds map
            bounds[var] = Interval(new_lb, new_ub)

    return bounds, monotonicity, convexity


def best_lower_bound(var, a, b, eps):
    """Returns the best lower bound between `a` and `b`.

    Parameters
    ----------
    var : Var
        the variable
    a : float or None
    b : float or None
    """
    if b is None:
        lb = a
    elif a is not None:
        lb = max(a, b)
    else:
        return None

    if (var.is_integer() or var.is_binary()) and lb is not None:
        if is_close(np.floor(lb), lb, atol=eps, rtol=0.0):
            return np.floor(lb)
        return np.ceil(lb)

    return lb


def best_upper_bound(var, a, b, eps):
    """Returns the best upper bound between `a` and `b`.

    Parameters
    ----------
    var : Var
        the variable
    a : float or None
    b : float or None
    """
    if b is None:
        ub = a
    elif a is not None:
        ub = min(a, b)
    else:
        return None

    if (var.is_integer() or var.is_binary()) and ub is not None:
        if is_close(np.ceil(ub), ub, atol=eps, rtol=0.0):
            return np.ceil(ub)
        return np.floor(ub)

    return ub
