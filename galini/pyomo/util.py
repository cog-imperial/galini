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

"""Utility functions for galini.pyomo."""

import pyomo.environ as pe
try:
    from pypopt.solver import PypoptDirectSolver
    pypopt_available = True
except ImportError:
    pypopt_available = False


def safe_set_bounds(var, lb, ub):
    safe_setlb(var, lb)
    safe_setub(var, ub)


def safe_setlb(var, lb):
    if lb is not None:
        lb = float(lb)
    var.setlb(lb)


def safe_setub(var, ub):
    if ub is not None:
        ub = float(ub)
    var.setub(ub)


def constraint_violation(constraint):
    if constraint.has_lb() and constraint.has_ub():
        raise ValueError('Constraint with lower and upper bound is not supported')
    if constraint.has_lb():
        return abs(min(constraint.lslack(), 0.0))
    assert constraint.has_ub()
    return abs(min(constraint.uslack(), 0.0))


def model_variables(model):
    """Return a list of variables in the model"""
    for variables in model.component_map(pe.Var, active=True).itervalues():
        for idx in variables:
            yield variables[idx]


def model_constraints(model):
    """Return a list of constraints in the model"""
    for cons in model.component_map(pe.Constraint, active=True).itervalues():
        for idx in cons:
            yield cons[idx]


def model_objectives(model):
    """Return a list of objectives in the model"""
    for obj in model.component_map(pe.Objective, active=True).itervalues():
        for idx in obj:
            yield obj[idx]


def instantiate_solver_with_options(solver_options):
    """Create a new Pyomo solver."""
    name = solver_options['name']
    options = solver_options['options']

    solver = pe.SolverFactory(name)
    for key, value in options.items():
        solver.options[key] = value

    solver._galini_meta = dict(
        (k, solver_options[k])
        for k in ['timelimit_option', 'relative_gap_option', 'absolute_gap_option', 'maxiter_option']
    )

    return solver


def update_solver_options(solver, timelimit=None, relative_gap=None, absolute_gap=None, maxiter=None):
    """Update the solver with the given timelimit and gaps."""
    assert hasattr(solver, '_galini_meta'), 'Create solver using instantiate_solver_with_options'

    meta = solver._galini_meta

    if timelimit is not None:
        timelimit_option = meta['timelimit_option']
        if timelimit_option:
            solver.options[timelimit_option] = timelimit

    if relative_gap is not None:
        relative_gap_option = meta['relative_gap_option']
        if relative_gap_option:
            solver.options[relative_gap_option] = relative_gap

    if absolute_gap is not None:
        absolute_gap_option = meta['absolute_gap_option']
        if absolute_gap_option:
            solver.options[absolute_gap_option] = absolute_gap

    if maxiter is not None:
        maxiter_option = meta['maxiter_option']
        if maxiter_option:
            solver.options[maxiter_option] = maxiter