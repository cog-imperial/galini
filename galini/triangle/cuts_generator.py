# Copyright 2019 Radu Baltean-Lugojan
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
"""Triangle Cuts generator."""
import numpy as np
from networkx import enumerate_all_cliques, from_numpy_matrix
from suspect.expression import ExpressionType
from galini.config import CutsGeneratorOptions, NumericOption
from galini.core import LinearExpression, SumExpression, QuadraticExpression
from galini.cuts import CutType, Cut, CutsGenerator
from galini.math import mc, is_close
from galini.logging import get_logger
from galini.util import solution_numerical_value


logger = get_logger(__name__)


class TriangleCutsGenerator(CutsGenerator):
    """
    Implements triangle cuts (with scaling at each B&B node, see Appendix A
    from 'Globally solving nonconvex quadratic programming problems with box
    constraints via integer programming methods',
    Bonami, Pierre and Gunluk, Oktay and Linderoth, Jeff,
    Mathematical Programming Computation, 1-50, 2018, Springer).
    """
    name = 'triangle'

    def __init__(self, galini, config):
        self._domain_eps = config['domain_eps']
        self._sel_size = config['selection_size']
        self._thres_tri_viol = config['thres_triangle_viol']
        self._min_tri_cuts = config['min_tri_cuts_per_round']
        self._max_tri_cuts = config['max_tri_cuts_per_round']
        self._cuts_relative_tolerance = config['convergence_relative_tolerance']

        # Problem info related to triangle cuts associated with every node of BabAlgorithm
        self.__problem_info_triangle = None
        self._nb_vars = 0   # number of variables in problem
        self._cut_outer_iteration = 0
        self._cut_round = 0
        self._lbs = None    # lower bounds
        self._ubs = None    # upper bounds
        self._dbs = None    # difference in bounds

    @staticmethod
    def cuts_generator_options():
        return CutsGeneratorOptions(TriangleCutsGenerator.name, [
            NumericOption('domain_eps',
                          default=1e-3,
                          description='Minimum domain length for each variable to consider cut on'),
            NumericOption('selection_size',
                          default=0.1,
                          description='Cut selection size as a % of all cuts or as absolute number of cuts'),
            NumericOption('thres_triangle_viol',
                          default=1e-7,
                          description='Violation threshold for separation of triangle inequalities'),
            NumericOption('min_tri_cuts_per_round',
                          default=5e3,
                          description='Min number of triangle cuts to be added to relaxation at each cut round'),
            NumericOption('max_tri_cuts_per_round',
                          default=10e3,
                          description='Max number of triangle cuts to be added to relaxation at each cut round'),
            NumericOption('convergence_relative_tolerance',
                          default=1e-3,
                          description='Termination criteria on lower bound improvement between '
                                      'two consecutive cut rounds <= relative_tolerance % of '
                                      'lower bound improvement from cut round'),
        ])

    def before_start_at_root(self, run_id, problem, relaxed_problem):
        self._nb_vars = problem.num_variables
        self._get_triangle_info(problem)
        self.before_start_at_node(run_id, problem, relaxed_problem)

    def after_end_at_root(self, run_id, problem, relaxed_problem, solution):
        self.after_end_at_node(run_id, problem, relaxed_problem, solution)

    def before_start_at_node(self, run_id, problem, relaxed_problem):
        self._lbs = problem.lower_bounds
        self._ubs = problem.upper_bounds
        self._dbs = [u - l for l, u in zip(self._lbs, self._ubs)]
        self._cut_round = 0

    def after_end_at_node(self, run_id, problem, relaxed_problem, solution):
        self._lbs = None
        self._ubs = None
        self._dbs = None

    def has_converged(self, state):
        """Termination criteria for cut generation loop.

        Termination criteria on lower bound improvement between two consecutive
        cut rounds <= relative_tolerance % of lower bound improvement from cut round.
        """
        if (state.first_solution is None or
            state.previous_solution is None or
            state.latest_solution is None):
            return False

        if is_close(state.latest_solution, state.previous_solution, atol=mc.epsilon):
            return True

        improvement = state.latest_solution - state.previous_solution
        lower_bound_improvement = state.latest_solution - state.first_solution
        if is_close(lower_bound_improvement, 0.0, atol=mc.epsilon):
            return True
        return (improvement / lower_bound_improvement) <= self._cuts_relative_tolerance

    def generate(self, run_id, problem, relaxed_problem, linear_problem, solution, tree, node):
        cuts = list(self._generate(run_id, problem, relaxed_problem, linear_problem, solution, tree, node))
        self._cut_round += 1
        self._cut_outer_iteration += 1
        return cuts

    def _generate(self, run_id, problem, _relaxed_problem, linear_problem, solution, tree, node):
        triple_cliques = self.__problem_info_triangle[1]
        rank_list_tri = self._get_triangle_violations(linear_problem, solution)
        # Remove non-violated constraints and sort by density first and then violation second as in manuscript
        rank_list_tri_viol = [el for el in rank_list_tri if el[2] >= self._thres_tri_viol]
        rank_list_tri_viol.sort(key=lambda tup: tup[2], reverse=True)

        # Determine number of triangle cuts to add (proportion/absolute with upper & lower thresholds)
        nb_cuts = int(np.floor(self._sel_size * len(rank_list_tri_viol))) \
            if self._sel_size <= 1 else int(np.floor(self._sel_size))
        max_tri_cuts = min(
            max(self._min_tri_cuts, nb_cuts),
            min(self._max_tri_cuts, len(rank_list_tri_viol)))
        max_tri_cuts = int(max_tri_cuts)
        l = self._lbs
        u = self._ubs
        d = self._dbs

        # Add all triangle cuts (ranked by violation) within selection size
        logger.debug(run_id, 'Adding {} cuts', max_tri_cuts)
        for ix in range(0, max_tri_cuts):
            ineq_type = rank_list_tri_viol[ix][1]
            i, j, k = triple_cliques[rank_list_tri_viol[ix][0]]
            xi, xj, xk = problem.variables[i], problem.variables[j], problem.variables[k]
            # Generate constraints for the 4 different triangle inequality types
            cut_lb = 0
            logger.debug(run_id, 'Cut {} is of type {}', ix, ineq_type)
            logger.debug(run_id, 'd[i] = {}, d[j] = {}, d[k] = {}', d[i], d[j], d[k])
            logger.debug(run_id, 'l[i] = {}, l[j] = {}, l[k] = {}', l[i], l[j], l[k])
            logger.debug(run_id, 'u[i] = {}, u[j] = {}, u[k] = {}', u[i], u[j], u[k])
            if is_close(d[i], 0.0, atol=mc.epsilon):
                logger.warning(run_id, 'Skip Cut {}, d[i] is zero', ix)
                continue

            if is_close(d[j], 0.0, atol=mc.epsilon):
                logger.warning(run_id, 'Skip Cut {}, d[j] is zero', ix)
                continue

            if is_close(d[k], 0.0, atol=mc.epsilon):
                logger.warning(run_id, 'Skip Cut {}, d[k] is zero', ix)
                continue

            if ineq_type == 3:
                sum_expr = SumExpression([
                    QuadraticExpression([xi, xj, xk], [xj, xk, xi],
                                        [1.0/d[i]/d[j], 1.0/d[j]/d[k], 1.0/d[k]/d[i]]),
                    LinearExpression([xi, xj, xk],
                                     [
                                        -1.0/d[i] -l[j]/d[i]/d[j] -l[k]/d[i]/d[k],
                                        -1.0/d[j] -l[i]/d[j]/d[i] -l[k]/d[j]/d[k],
                                        -1.0/d[k] -l[i]/d[i]/d[k] -l[j]/d[j]/d[k]
                                     ],
                                     +l[i]*l[j]/d[i]/d[j] +l[i]*l[k]/d[i]/d[k] +l[j]*l[k]/d[j]/d[k]
                                     +l[i]/d[i] +l[j]/d[j] +l[k]/d[k])
                ])
                cut_lb = -1.0
            else:
                if ineq_type == 0:
                    sum_expr = SumExpression([
                        QuadraticExpression([xi, xj, xk], [xj, xk, xi],
                                            [-1.0/d[i]/d[j], 1.0/d[j]/d[k], -1.0/d[k]/d[i]]),
                        LinearExpression([xi, xj, xk],
                                         [
                                            1.0/d[i] +l[j]/d[i]/d[j] +l[k]/d[i]/d[k],
                                                    +l[i]/d[j]/d[i] -l[k]/d[j]/d[k],
                                                    +l[i]/d[i]/d[k] -l[j]/d[j]/d[k]
                                         ],
                                         -l[i]*l[j]/d[i]/d[j] - l[i]*l[k]/d[i]/d[k] + l[j]*l[k]/d[j]/d[k] -l[i]/d[i])
                    ])
                elif ineq_type == 1:
                    sum_expr = SumExpression([
                        QuadraticExpression([xi, xj, xk], [xj, xk, xi],
                                            [-1.0/d[i]/d[j], -1.0/d[j]/d[k], 1.0/d[k]/d[i]]),
                        LinearExpression([xi, xj, xk],
                                         [
                                                    +l[j]/d[i]/d[j] -l[k]/d[i]/d[k],
                                            1.0/d[j] +l[i]/d[j]/d[i] +l[k]/d[j]/d[k],
                                                    -l[i]/d[i]/d[k] +l[j]/d[j]/d[k]
                                         ],
                                         -l[i]*l[j]/d[i]/d[j] +l[i]*l[k]/d[i]/d[k] - l[j]*l[k]/d[j]/d[k] -l[j]/d[j])
                    ])
                elif ineq_type == 2:
                    sum_expr = SumExpression([
                        QuadraticExpression([xi, xj, xk], [xj, xk, xi],
                                            [1.0/d[i]/d[j], -1.0/d[j]/d[k], -1.0/d[k]/d[i]]),
                        LinearExpression([xi, xj, xk],
                                         [
                                                    -l[j]/d[i]/d[j] +l[k]/d[i]/d[k],
                                                    -l[i]/d[j]/d[i] +l[k]/d[j]/d[k],
                                            1.0/d[k] +l[i]/d[i]/d[k] +l[j]/d[j]/d[k]
                                         ],
                                        +l[i]*l[j]/d[i]/d[j] -l[i]*l[k]/d[i]/d[k] - l[j]*l[k]/d[j]/d[k] - l[k]/d[k])
                    ])

            cut_name = 'triangle_cut_{}_{}_{}_{}'.format(
                self._cut_outer_iteration, self._cut_round, ix, ineq_type
            )
            yield Cut(CutType.LOCAL, cut_name, sum_expr, cut_lb, None)

    def _get_triangle_info(self, problem):
        adj_mat = self._get_adjacency_matrix(problem)
        triple_cliques = []
        for clique in enumerate_all_cliques(from_numpy_matrix(adj_mat)):
            if len(clique) < 3:
                continue
            elif len(clique) == 3:
                triple_cliques.append(clique)
            else:
                break
        rank_list_tri = [0] * 4 * len(triple_cliques)
        for idx_clique, clique in enumerate(triple_cliques):
            for tri_cut_type in range(4):
                rank_list_tri[idx_clique * 4 + tri_cut_type] = [idx_clique, tri_cut_type, 0]
        self.__problem_info_triangle = (rank_list_tri, triple_cliques)

    def _get_adjacency_matrix(self, problem):
        adj_mat = np.zeros((self._nb_vars, self._nb_vars))
        vars_dict = dict([(v.name, v_idx) for v_idx, v in enumerate(problem.variables)])
        for constraint in [*[problem.objective], *problem.constraints]:
            root_expr = constraint.root_expr
            quadratic_expr = None
            if root_expr.expression_type == ExpressionType.Quadratic:
                quadratic_expr = root_expr
            elif root_expr.expression_type == ExpressionType.Sum:
                assert len(root_expr.children) == 2
                a, b = root_expr.children
                if a.expression_type == ExpressionType.Quadratic:
                    assert b.expression_type == ExpressionType.Linear
                    quadratic_expr = a
                if b.expression_type == ExpressionType.Quadratic:
                    assert a.expression_type == ExpressionType.Linear
                    quadratic_expr = b
            if quadratic_expr is not None:
                for term in quadratic_expr.terms:
                    if term.coefficient != 0:
                        adj_mat[vars_dict[term.var1.name], vars_dict[term.var2.name]] = 1
                        adj_mat[vars_dict[term.var2.name], vars_dict[term.var1.name]] = 1
        return adj_mat

    def _get_triangle_violations(self, problem, solution):
        rank_list_tri, triple_cliques = self.__problem_info_triangle
        lifted_mat = self._get_lifted_mat_values(problem, solution)
        # Evaluate violations for all valid triangle cliques and cut types
        x_vals = [0] * 3
        x_vals_scaled = [0] * 3
        l = self._lbs
        u = self._ubs
        d = self._dbs
        for idx_clique, cl in enumerate(triple_cliques):
            # If the domain of any variables is very small, don't consider cut
            if any(d[i] <= self._domain_eps for i in cl):
                continue
            for idx, var_idx in enumerate(cl):
                x_vals[idx] = solution_numerical_value(
                    solution.variables[var_idx], l[var_idx], u[var_idx]
                )
                x_vals_scaled[idx] = (x_vals[idx] - l[var_idx])/d[var_idx]
            x01 = (lifted_mat[cl[0], cl[1]] - l[cl[0]]*x_vals[1] - l[cl[1]]*x_vals[0] + l[cl[0]]*l[cl[1]])\
                  /d[cl[0]]/d[cl[1]]
            x02 = (lifted_mat[cl[0], cl[2]] - l[cl[0]]*x_vals[2] - l[cl[2]]*x_vals[0] + l[cl[0]]*l[cl[2]])\
                  /d[cl[0]]/d[cl[2]]
            x12 = (lifted_mat[cl[1], cl[2]] - l[cl[1]]*x_vals[2] - l[cl[2]]*x_vals[1] + l[cl[1]]*l[cl[2]])\
                  /d[cl[1]]/d[cl[2]]
            rank_list_tri[idx_clique * 4][2] = x01 + x02 - x12 - x_vals_scaled[0]
            rank_list_tri[idx_clique * 4 + 1][2] = x01 - x02 + x12 - x_vals_scaled[1]
            rank_list_tri[idx_clique * 4 + 2][2] = -x01 + x02 + x12 - x_vals_scaled[2]
            rank_list_tri[idx_clique * 4 + 3][2] = -x01 - x02 - x12 + sum(x_vals_scaled) - 1
        return rank_list_tri

    def _get_lifted_mat_values(self, problem, solution):
        # Build matrix of lifted X values
        nb_vars = self._nb_vars
        lifted_mat = np.zeros((nb_vars, nb_vars))
        for var_sol in solution.variables[nb_vars:problem.num_variables]:
            var = problem.variable(var_sol.name)
            if not var.is_auxiliary:
                continue
            var1 = var.reference.var1
            var2 = var.reference.var2
            lifted_mat[var1.idx, var2.idx] = var_sol.value
            lifted_mat[var2.idx, var1.idx] = var_sol.value
        return lifted_mat
