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
"""Generic Branch & Bound solver."""
import pyomo.environ as pe
from pyomo.core.kernel.component_set import ComponentSet
from pyomo.core.expr.current import identify_variables
import numpy as np
from coramin.domain_reduction.obbt import perform_obbt
from coramin.relaxations.auto_relax import relax
from galini.logging import get_logger
from galini.config import (
    SolverOptions,
    NumericOption,
    IntegerOption,
    EnumOption,
)
from galini.solvers import Solver
from galini.cuts import CutsGeneratorsRegistry
from galini.bab.branch_and_cut import BranchAndCutAlgorithm
from galini.util import print_problem


logger = get_logger(__name__)


class BranchAndBoundSolver(Solver):
    name = 'bab'

    description = 'Generic Branch & Bound solver.'

    @staticmethod
    def solver_options():
        return SolverOptions(BranchAndBoundSolver.name, [
            NumericOption('tolerance', default=1e-8),
            NumericOption('relative_tolerance', default=1e-8),
            IntegerOption('node_limit', default=100000000),
            IntegerOption('fbbt_maxiter', default=10),
            IntegerOption('obbt_simplex_maxiter', default=1000),
            BranchAndCutAlgorithm.algorithm_options(),
        ])

    def before_solve(self, model, problem):
        relaxed_model = relax(model)

        for obj in relaxed_model.component_data_objects(ctype=pe.Objective):
            relaxed_model.del_component(obj)

        solver = pe.SolverFactory('cplex_persistent')
        solver.set_instance(relaxed_model)
        obbt_simplex_maxiter = self.config['obbt_simplex_maxiter']
        solver._solver_model.parameters.simplex.limits.iterations.set(obbt_simplex_maxiter)
        # collect variables in nonlinear constraints
        nonlinear_variables = ComponentSet()
        for constraint in model.component_data_objects(ctype=pe.Constraint):
            # skip linear constraint
            if constraint.body.polynomial_degree() == 1:
                continue

            for var in identify_variables(constraint.body, include_fixed=False):
                nonlinear_variables.add(var)

        relaxed_vars = [getattr(relaxed_model, v.name) for v in nonlinear_variables]

        for var in relaxed_vars:
            var.domain = pe.Reals

        logger.info(0, 'Performaning OBBT on {} variables: {}', len(relaxed_vars), [v.name for v in relaxed_vars])

        try:
            result = perform_obbt(relaxed_model, solver, relaxed_vars)
            if result is None:
                return

            for v, new_lb, new_ub in zip(relaxed_vars, *result):
                vv = problem.variable_view(v.name)
                vv.set_lower_bound(_safe_lb(vv.domain, new_lb, vv.lower_bound()))
                vv.set_upper_bound(_safe_ub(vv.domain, new_ub, vv.upper_bound()))
        except Exception as ex:
            logger.warning(0, 'Error performing OBBT: {}', ex)
            return

    def actual_solve(self, problem, run_id, **kwargs):
        algo = BranchAndCutAlgorithm(self.galini)
        return algo.solve(problem, run_id=run_id)




def _safe_lb(domain, a, b):
    if b is None:
        lb = a
    else:
        lb = max(a, b)

    if domain.is_integer():
        return np.ceil(lb)

    return lb


def _safe_ub(domain, a, b):
    if b is None:
        ub = a
    else:
        ub = min(a, b)

    if domain.is_integer():
        return np.floor(ub)

    return ub
