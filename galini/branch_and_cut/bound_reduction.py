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
import numpy as np
import pyomo.environ as pe
from coramin.relaxations.iterators import relaxation_data_objects
from pyomo.core.kernel.component_set import ComponentSet
from suspect.fbbt import perform_fbbt
from suspect.interval import Interval
from suspect.propagation import propagate_special_structure

from galini.math import is_close
from galini.pyomo import safe_setlb, safe_setub
from galini.timelimit import (
    current_time,
    seconds_elapsed_since,
    timeout,
)

coramin_logger = coramin_obbt.logger  # pylint: disable=invalid-name
coramin_logger.disabled = True


def perform_obbt_on_model(model, linear_model, upper_bound, timelimit, simplex_maxiter, mc):
    """Perform OBBT on Pyomo model using Coramin.

    Parameters
    ----------
    model : ConcreteModel
        the pyomo concrete model
    linear_model : ConcreteModel
        the linear relaxation of model
    upper_bound : float or None
        the objective value upper bound, if known
    timelimit : int
        a timelimit, in seconds
    simplex_maxiter : int
        the maximum number of simplex iterations
    mc : MathContext
        GALINI math context
    """
    obbt_start_time = current_time()

    for var in linear_model.component_data_objects(ctype=pe.Var):
        var.domain = pe.Reals

    solver = pe.SolverFactory('cplex')
    # TODO(fra): make this non-cplex specific
    solver.options['parameters simplex limits iterations'] = simplex_maxiter

    # collect variables in nonlinear constraints
    nonlinear_variables = ComponentSet()
    for relaxation in relaxation_data_objects(linear_model, active=True, descend_into=True):
        for var in relaxation.get_rhs_vars():
            # Coramin will complain about variables that are fixed
            # Note: Coramin uses an hard-coded 1e-6 tolerance
            if not var.has_lb() or not var.has_ub():
                nonlinear_variables.add(var)
            else:
                if not np.abs(var.ub - var.lb) < mc.epsilon:
                    nonlinear_variables.add(var)

    time_left = timelimit - seconds_elapsed_since(obbt_start_time)
    nonlinear_variables = list(nonlinear_variables)
    with timeout(time_left, 'Timeout in OBBT'):
        result = coramin_obbt.perform_obbt(
            linear_model, solver,
            varlist=nonlinear_variables,
            objective_bound=upper_bound,
            warning_threshold=mc.epsilon
        )

    if result is None:
        return

    new_bounds = pe.ComponentMap()

    eps = mc.epsilon

    for var, new_lb, new_ub in zip(nonlinear_variables, *result):
        new_lb = best_lower_bound(var, new_lb, var.lb, eps)
        new_ub = best_upper_bound(var, new_ub, var.ub, eps)
        new_bounds[var] = (new_lb, new_ub)
        var.setlb(new_lb)
        var.setub(new_ub)
        original_var = model.find_component(var.getname(fully_qualified=True))
        original_var.setlb(new_lb)
        original_var.setub(new_ub)

    # Rebuild relaxations since bounds changed
    for relaxation in relaxation_data_objects(linear_model, active=True, descend_into=True):
        relaxation.rebuild()

    return new_bounds


def perform_fbbt_on_model(model, tree, node, maxiter, eps):
    fbbt_start_time = current_time()

    objective_upper_bound = None
    if tree.upper_bound is not None:
        objective_upper_bound = tree.upper_bound

    branching_variable = None
    if not node.storage.is_root:
        branching_variable = node.storage.branching_variable
    bounds = perform_fbbt(
        model,
        max_iter=maxiter,
        #timelimit=self.fbbt_timelimit,
        #objective_upper_bound=objective_upper_bound,
        #branching_variable=branching_variable,
    )

    monotonicity, convexity = \
        propagate_special_structure(model, bounds)

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
            safe_setlb(var, new_lb)
            safe_setub(var, new_ub)
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
