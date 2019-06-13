# Copyright 2019 Francesco Ceccon
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

"""Outer-Approximation cuts generator."""
import numpy as np
from contextlib import contextmanager
from galini.config import CutsGeneratorOptions
from galini.core import LinearExpression, Domain
from galini.cuts import CutType, Cut, CutsGenerator
from galini.logging import get_logger


logger = get_logger(__name__)


@contextmanager
def fixed_integer_variables(run_id, problem, solution):
    """Context manager to fix integer variables."""
    for i, v in enumerate(problem.variables):
        v_view = problem.variable_view(v)
        if v_view.domain != Domain.REAL:
            value = solution.variables[i].value
            logger.debug(run_id, 'Fix {} to {}', v.name, value)
            v_view.fix(value)

    try:
        yield problem
    finally:
        for v in problem.variables:
            problem.unfix(v)



class OuterApproximationCutsGenerator(CutsGenerator):
    """Implement Outer-Appoximation cuts for Convex MINLP problems.

    References
    ----------
    [0] Bonami P., Biegler L. T., Conn A. R., Cornuéjols G., Grossmann I. E., Laird C. D.,
        Lee J, Lodi A., Margot F., Sawaya N., Wächter A. (2008).
        An algorithmic framework for convex mixed integer nonlinear programs.
        Discrete Optimization
        https://doi.org/10.1016/J.DISOPT.2006.10.011
    [1] Duran, M. A., Grossmann, I. E. (1986).
        An outer-approximation algorithm for a class of mixed-integer nonlinear programs.
        Mathematical Programming
        https://doi.org/10.1007/BF02592064
    """

    name = 'outer_approximation'

    def __init__(self, galini, config):
        super().__init__(galini, config)
        self._reset_node_local_storage()
        self._nlp_solver = galini.instantiate_solver('ipopt')

    def _reset_node_local_storage(self):
        self._round = 0

    @staticmethod
    def cuts_generator_options():
        return CutsGeneratorOptions(OuterApproximationCutsGenerator.name, [
        ])

    def before_start_at_root(self, run_id, problem, relaxed_problem):
        self._reset_node_local_storage()

    def after_end_at_root(self, run_id, problem, relaxed_problem, solution):
        pass

    def before_start_at_node(self, run_id, problem, relaxed_problem):
        self._reset_node_local_storage()

    def after_end_at_node(self, run_id, problem, relaxed_problem, solution):
        pass

    def generate(self, run_id, problem, relaxed_problem, mip_solution, tree, node):
        logger.debug(run_id, 'Starting cut round={}', self._round)
        logger.debug(run_id, 'Compute points for non integer variables')
        f_x_solution = self._solve_nlp_with_integer_fixed(run_id, relaxed_problem, mip_solution)
        logger.debug(run_id, 'Solving NLP returned {}', f_x_solution)
        assert f_x_solution.status.is_success()

        x_k = [v.value for v in f_x_solution.variables[:relaxed_problem.num_variables]]

        logger.debug(run_id, 'Computing derivatives at point x_k={}', x_k)
        fg = self._compute_derivatives(run_id, relaxed_problem, x_k)
        fg_x = fg.forward(0, x_k)
        logger.debug(run_id, 'Objectives/Constraints value at x_k={}', fg_x)

        x = relaxed_problem.variables

        num_objectives = relaxed_problem.num_objectives
        num_constraints = relaxed_problem.num_constraints

        w = np.zeros(num_objectives + num_constraints)

        logger.debug(run_id, 'Generating {} objective cuts', num_objectives)
        objective_cuts = [
            self._generate_cut(i, objective, x, x_k, fg, fg_x, w, True)
            for i, objective in enumerate(relaxed_problem.objectives)
        ]

        logger.debug(run_id, 'Generating {} constraints cuts', num_constraints)
        constraint_cuts = [
            self._generate_cut(num_objectives + i, constraint, x, x_k, fg, fg_x, w, False)
            for i, constraint in enumerate(relaxed_problem.constraints)
        ]

        self._round += 1
        return objective_cuts + constraint_cuts

    def _compute_derivatives(self, run_id, relaxed_problem, x_k):
        f_idx = [f.root_expr.idx for f in relaxed_problem.objectives]
        g_idx = [g.root_expr.idx for g in relaxed_problem.constraints]
        return relaxed_problem.expression_tree_data().eval(x_k, f_idx + g_idx)

    def _solve_nlp_with_integer_fixed(self, run_id, relaxed_problem, mip_solution):

        with fixed_integer_variables(run_id, relaxed_problem, mip_solution) as problem:
            nlp_solution = self._nlp_solver.solve(problem)
            assert nlp_solution.status.is_success()
            return nlp_solution

    def _generate_cut(self, i, constraint, x, x_k, fg, g_x, w, is_objective):
        w[i] = 1.0
        d_fg = fg.reverse(1, w)
        w[i] = 0.0
        cut_name = '_outer_approximation_{}_{}_round_{}'.format(i, constraint.name, self._round)

        if not is_objective:
            return Cut(
                CutType.LOCAL,
                cut_name,
                LinearExpression(x, d_fg, -np.dot(d_fg, x_k) + g_x[i]),
                constraint.lower_bound,
                constraint.upper_bound,
            )

        return Cut(
            CutType.LOCAL,
            cut_name,
            LinearExpression(x, d_fg, -np.dot(d_fg, x_k) + g_x[i]),
            None,
            None,
            is_objective=True,
        )
