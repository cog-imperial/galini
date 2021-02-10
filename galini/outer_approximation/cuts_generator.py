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

from pyomo.core.expr.calculus.diff_with_pyomo import _diff_SumExpression, _diff_map
from suspect.pyomo.quadratic import QuadraticExpression
import pyomo.environ as pe
from pyomo.core.expr.calculus.derivatives import differentiate
from coramin.relaxations import relaxation_data_objects
from coramin.utils.coramin_enums import RelaxationSide


from galini.config import CutsGeneratorOptions, NumericOption
from galini.cuts import CutsGenerator
from galini.math import almost_ge, almost_le


_diff_map[QuadraticExpression] = _diff_SumExpression


def get_deviation_adjusted_by_side(aux_var, rhs_expr, side):
    if side == RelaxationSide.UNDER or side == RelaxationSide.BOTH:
        return -min(0, pe.value(aux_var) - pe.value(rhs_expr))
    return -min(0, pe.value(rhs_expr) - pe.value(aux_var))


class OuterApproximationCutsGenerator(CutsGenerator):
    """Implement Outer-Approximation cuts for Convex Expressions."""

    name = 'outer_approximation'

    description = 'outer approximation cuts'

    def __init__(self, galini, config):
        super().__init__(galini, config)
        self.galini = galini
        self.logger = galini.get_logger(__name__)

        self._convex_relaxations_map = None
        self._prefix = 'outer_approximation'
        self._counter = 0

        self.convergence_relative_tol = config['convergence_relative_tol']
        self.threshold = config['deviation_threshold']

    @staticmethod
    def cuts_generator_options():
        """Quadratic Outer Approximation cuts generator options"""
        return CutsGeneratorOptions(
            OuterApproximationCutsGenerator.name, [
            NumericOption('deviation_threshold', default=1e-5),
            NumericOption('convergence_relative_tol', default=1e-3),
        ])

    def _check_if_problem_is_nolinear(self, relaxed_problem):
        self._convex_relaxations_map = pe.ComponentMap()
        for relaxation in relaxed_problem.galini_nonlinear_relaxations:
            is_convex = (
                relaxation.is_rhs_convex() and
                relaxation.relaxation_side in [RelaxationSide.BOTH, RelaxationSide.UNDER]
            ) or (
                relaxation.is_rhs_concave() and
                relaxation.relaxation_side in [RelaxationSide.BOTH, RelaxationSide.OVER]
            )
            if is_convex:
                rhs_expr = relaxation.get_rhs_expr()
                rhs_vars = relaxation.get_rhs_vars()
                aux_var = relaxation.get_aux_var()
                rel_expr = rhs_expr - aux_var
                rel_vars = rhs_vars + [aux_var]
                self._convex_relaxations_map[relaxation] = (rel_expr, rel_vars)

    def before_start_at_root(self, problem, relaxed_problem):
        self._check_if_problem_is_nolinear(relaxed_problem)

    def after_end_at_root(self, problem, relaxed_problem, solution):
        pass

    def before_start_at_node(self, problem, relaxed_problem):
        self._check_if_problem_is_nolinear(relaxed_problem)

    def after_end_at_node(self, problem, relaxed_problem, solution):
        pass

    def has_converged(self, state):
        if not self._convex_relaxations_map:
            return True
        return False

    def generate(self, problem, relaxed_problem, mip_solution, tree, node):
        if not self._convex_relaxations_map:
            return []

        cuts = []
        for relaxation, (rel_expr, rel_vars) in self._convex_relaxations_map.items():
            aux_var = relaxation.get_aux_var()
            rhs_expr = relaxation.get_rhs_expr()
            side = relaxation.relaxation_side
            deviation = get_deviation_adjusted_by_side(aux_var, rhs_expr, side)
            if deviation < self.threshold:
                continue
            diff_map = differentiate(rel_expr, wrt_list=rel_vars)
            cut_expr = pe.value(rel_expr) + sum(diff_map[i] * (v - pe.value(v)) for i, v in enumerate(rel_vars))
            relaxation_side = relaxation.relaxation_side
            if relaxation_side == RelaxationSide.UNDER:
                cut_ineq = cut_expr <= 0
            elif relaxation_side == RelaxationSide.OVER:
                cut_ineq = cut_expr >= 0
            else:
                assert relaxation_side == RelaxationSide.BOTH
                cut_ineq = cut_expr <= 0

            cuts.append(cut_ineq)

        return cuts
