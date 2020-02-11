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

import numpy as np

from galini.config import CutsGeneratorOptions, NumericOption
from galini.cuts import CutsGenerator
from galini.expression_relaxation.expression_relaxation import RelaxationSide
from galini.logging import get_logger
from galini.math import almost_ge, almost_le
from galini.outer_approximation.shared import (
    problem_is_linear, mip_variable_value, generate_cut
)

logger = get_logger(__name__)


class OuterApproximationCutsGenerator(CutsGenerator):
    """Implement Outer-Approximation cuts for Convex Expressions."""

    name = 'outer_approximation'

    def __init__(self, galini, config):
        super().__init__(galini, config)
        self.galini = galini

        self._relaxation_is_linear = False
        self._prefix = 'outer_approximation'
        self._counter = 0

        self.convergence_relative_tol = config['convergence_relative_tol']
        self.threshold = config['threshold']

    @staticmethod
    def cuts_generator_options():
        """Quadratic Outer Approximation cuts generator options"""
        return CutsGeneratorOptions(
            OuterApproximationCutsGenerator.name, [
            NumericOption('threshold', default=1e-3),
            NumericOption('convergence_relative_tol', default=1e-3),
        ])

    def _check_if_problem_is_nolinear(self, relaxed_problem):
        self._relaxation_is_linear = problem_is_linear(relaxed_problem)

    def before_start_at_root(self, run_id, problem, relaxed_problem):
        self._check_if_problem_is_nolinear(relaxed_problem)

    def after_end_at_root(self, run_id, problem, relaxed_problem, solution):
        pass

    def before_start_at_node(self, run_id, problem, relaxed_problem):
        self._check_if_problem_is_nolinear(relaxed_problem)

    def after_end_at_node(self, run_id, problem, relaxed_problem, solution):
        pass

    def has_converged(self, state):
        if self._relaxation_is_linear:
            return True
        return False

    def generate(self, run_id, problem, relaxed_problem, linear_problem,
                 mip_solution, tree, node):

        if self._relaxation_is_linear:
            return []

        nonlinear_constraints = [
            constraint
            for constraint in relaxed_problem.constraints
            if constraint.root_expr.polynomial_degree() > 1
        ]

        linearized_nonlinear_constraints = [
            linear_problem.constraint(constraint.name)
            for constraint in nonlinear_constraints
        ]

        if not nonlinear_constraints:
            return []

        nonlinear_constraints_idx = [
            con.root_expr.idx for con in nonlinear_constraints
        ]

        linearized_nonlinear_constraints_idx = [
            con.root_expr.idx for con in linearized_nonlinear_constraints
        ]

        mip_values_relaxed_vars = [
            mip_variable_value(linear_problem, v)
            for v in mip_solution.variables[:relaxed_problem.num_variables]
        ]

        mip_values_vars = [
            mip_variable_value(linear_problem, v)
            for v in mip_solution.variables
        ]

        nonlinear_expr_eval = \
            relaxed_problem.expression_tree_data()\
                .eval(mip_values_relaxed_vars, nonlinear_constraints_idx)

        linearized_nonlinear_expr_eval = \
            linear_problem.expression_tree_data()\
                .eval(mip_values_vars, linearized_nonlinear_constraints_idx)

        nonlinear_constraints_value = \
            np.array(nonlinear_expr_eval.forward(0, mip_values_relaxed_vars))

        linearized_nonlinear_constraints_value = \
            np.array(linearized_nonlinear_expr_eval.forward(0, mip_values_vars))


        w = np.zeros_like(nonlinear_constraints_value)
        x_k = mip_values_relaxed_vars
        g_x = nonlinear_constraints_value

        cuts = [
            self._generate_cut_if_violated(i, constraint, relaxed_problem.variables, w, x_k, nonlinear_expr_eval, nonlinear_constraints_value)
            for i, constraint in enumerate(nonlinear_constraints)
        ]
        return [c for c in cuts if c is not None]

    def _generate_cut_if_violated(self, i, constraint, variables, w, x_k, fg, g_x):
        cons_lb = constraint.lower_bound
        cons_ub = constraint.upper_bound
        constraint_x = g_x[i]

        self._counter += 1

        original_side = constraint.metadata.get('original_side', None)
        if cons_lb is None:
            # g(x) <= c
            above_threshold = not almost_le(
                constraint_x,
                cons_ub,
                atol=self.threshold,
            )
            if above_threshold:
                return generate_cut(
                    self._prefix, self._counter, i, constraint, variables,
                    w, x_k, fg, g_x
                )
        elif cons_ub is None:
            # c <= g(x)
            below_threshold = not almost_ge(
                constraint_x,
                cons_lb,
                atol=self.threshold,
            )
            if below_threshold:
                return generate_cut(
                    self._prefix, self._counter, i, constraint, variables,
                    w, x_k, fg, g_x
                )
        else:
            above_threshold = below_threshold = False
            if original_side == RelaxationSide.UNDER:
                above_threshold = not almost_le(
                    constraint_x,
                    cons_ub,
                    atol=self.threshold,
                )
            elif original_side == RelaxationSide.OVER:
                below_threshold = not almost_ge(
                    constraint_x,
                    cons_lb,
                    atol=self.threshold,
                )

            if above_threshold or below_threshold:
                return generate_cut(
                    self._prefix, self._counter, i, constraint, variables,
                    w, x_k, fg, g_x, original_side=original_side,
                )
