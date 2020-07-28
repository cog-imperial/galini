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

import pyomo.environ as pe
import numpy as np
from networkx import enumerate_all_cliques, from_numpy_matrix, from_edgelist
from suspect.expression import ExpressionType
from galini.config import CutsGeneratorOptions, NumericOption
from galini.cuts import CutType, Cut, CutsGenerator
from galini.math import is_close
from galini.quantities import relative_bound_improvement

from coramin.relaxations import relaxation_data_objects
from coramin.relaxations.mccormick import PWMcCormickRelaxation
from coramin.relaxations.univariate import PWXSquaredRelaxation
from coramin.utils.coramin_enums import RelaxationSide


class TriangleCutsGenerator(CutsGenerator):
    """
    Implements triangle cuts (with scaling at each B&B node, see Appendix A
    from 'Globally solving nonconvex quadratic programming problems with box
    constraints via integer programming methods',
    Bonami, Pierre and Gunluk, Oktay and Linderoth, Jeff,
    Mathematical Programming Computation, 1-50, 2018, Sprinvger).
    """
    name = 'triangle'

    def __init__(self, galini, config):
        super().__init__(galini, config)
        self.galini = galini
        self.logger = galini.get_logger(__name__)

        self._domain_eps = config['domain_eps']
        self._sel_size = config['selection_size']
        self._thres_tri_viol = config['thres_triangle_viol']
        self._min_tri_cuts = config['min_tri_cuts_per_round']
        self._max_tri_cuts = config['max_tri_cuts_per_round']
        self._cuts_relative_tolerance = config['convergence_relative_tolerance']

        # Problem info related to triangle cuts associated with every node of BabAlgorithm
        self._cut_outer_iteration = 0
        self._cut_round = 0
        self._lower_bounds = None    # lower bounds
        self._upper_bounds = None    # upper bounds
        self._domains = None         # difference in bounds
        self._aux_vars = None
        self._clique_with_rank = None

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

    def before_start_at_root(self, problem, relaxed_problem):
        self.before_start_at_node(problem, relaxed_problem)

    def after_end_at_root(self, problem, relaxed_problem, solution):
        self.after_end_at_node(problem, relaxed_problem, solution)

    def before_start_at_node(self, problem, relaxed_problem):
        self._compute_clique_ranks(relaxed_problem)
        self._cut_round = 0

    def after_end_at_node(self, problem, relaxed_problem, solution):
        self._lower_bounds = None
        self._upper_bounds = None
        self._domains = None
        self._aux_vars = None

    def has_converged(self, state):
        """Termination criteria for cut generation loop."""
        if not self._clique_with_rank:
            return True

        if state.first_solution is None or state.previous_solution is None or state.latest_solution is None:
            return False

        return relative_bound_improvement(
            state.first_solution,
            state.previous_solution,
            state.latest_solution,
            self.galini.mc
        ) <= self._cuts_relative_tolerance

    def generate(self, problem, relaxed_problem, mip_solution, tree, node):
        self._cut_round += 1
        self._cut_outer_iteration += 1

        clique_with_rank = self._get_triangle_violations()
        # Remove non-violated constraints and sort by density first and then violation second as in manuscript
        clique_with_rank = [
            clique for clique in clique_with_rank if clique[2] >= self._thres_tri_viol
        ]
        clique_with_rank.sort(key=lambda clique: clique[2], reverse=True)

        # Determine number of triangle cuts to add (proportion/absolute with upper & lower thresholds)
        if self._sel_size <= 1:
            nb_cuts = int(np.floor(self._sel_size * len(clique_with_rank)))
        else:
            nb_cuts = int(np.floor(self._sel_size))

        max_tri_cuts = min(
            max(self._min_tri_cuts, nb_cuts),
            min(self._max_tri_cuts, len(clique_with_rank)))
        max_tri_cuts = int(max_tri_cuts)

        lb = self._lower_bounds
        ub = self._upper_bounds
        dom = self._domains
        mc = self.galini.mc

        # Add all triangle cuts (ranked by violation) within selection size
        self.logger.debug('Adding {} cuts', max_tri_cuts)

        cuts = []
        for ix, (clique, ineq_type, viol) in enumerate(clique_with_rank[:max_tri_cuts]):
            xi, xj, xk = clique
            # Generate constraints for the 4 different triangle inequality types
            self.logger.debug('Cut {} is of type {}', ix, ineq_type)
            self.logger.debug('d[i] = {}, d[j] = {}, d[k] = {}', dom[xi], dom[xj], dom[xk])
            self.logger.debug('l[i] = {}, l[j] = {}, l[k] = {}', lb[xi], lb[xj], lb[xk])
            self.logger.debug('u[i] = {}, u[j] = {}, u[k] = {}', ub[xi], ub[xj], ub[xk])

            if is_close(dom[xi], 0.0, atol=mc.epsilon):
                self.logger.warning('Skip Cut {}, d[i] is zero', ix)
                continue

            if is_close(dom[xj], 0.0, atol=mc.epsilon):
                self.logger.warning('Skip Cut {}, d[j] is zero', ix)
                continue

            if is_close(dom[xk], 0.0, atol=mc.epsilon):
                self.logger.warning('Skip Cut {}, d[k] is zero', ix)
                continue

            xi_xj = self._aux_vars[id(xi), id(xj)]
            xi_xk = self._aux_vars[id(xi), id(xk)]
            xj_xk = self._aux_vars[id(xj), id(xk)]

            if ineq_type == 0:
                cut_expr = (
                    xi_xj*(-1.0/dom[xi]/dom[xj]) +
                    xj_xk*(+1.0/dom[xj]/dom[xk]) +
                    xi_xk*(-1.0/dom[xk]/dom[xi])
                ) + (
                    xi*(1.0/dom[xi] + lb[xj]/dom[xi]/dom[xj] + lb[xk]/dom[xi]/dom[xk]) +
                    xj*(lb[xi]/dom[xj]/dom[xi] - lb[xk]/dom[xj]/dom[xk]) +
                    xk*(lb[xi]/dom[xi]/dom[xk] - lb[xj]/dom[xj]/dom[xk])
                ) + (
                    -lb[xi]*lb[xj]/dom[xi]/dom[xj] +
                    -lb[xi]*lb[xk]/dom[xi]/dom[xk] +
                    +lb[xj]*lb[xk]/dom[xj]/dom[xk] +
                    -lb[xi]/dom[xi]
                )
                cuts.append(cut_expr >= 0)
            elif ineq_type == 1:
                cut_expr = (
                    xi_xj*(-1.0/dom[xi]/dom[xj]) +
                    xj_xk*(-1.0/dom[xj]/dom[xk]) +
                    xi_xk*(+1.0/dom[xk]/dom[xi])
                ) + (
                    xi*(lb[xj]/dom[xi]/dom[xj] - lb[xk]/dom[xi]/dom[xk]) +
                    xj*(1.0/dom[xj] + lb[xi]/dom[xj]/dom[xi] + lb[xk]/dom[xj]/dom[xk]) +
                    xk*(-lb[xi]/dom[xi]/dom[xk] + lb[xj]/dom[xj]/dom[xk])
                ) + (
                    -lb[xi]*lb[xj]/dom[xi]/dom[xj] +
                    +lb[xi]*lb[xk]/dom[xi]/dom[xk] +
                    -lb[xj]*lb[xk]/dom[xj]/dom[xk] +
                    -lb[xj]/dom[xj]
                )
                cuts.append(cut_expr >= 0)
            elif ineq_type == 2:
                cut_expr = (
                    xi_xj*(1.0/dom[xi]/dom[xj]) +
                    xj_xk*(-1.0/dom[xj]/dom[xk]) +
                    xi_xk*(-1.0/dom[xk]/dom[xi])
                ) + (
                    xi*(-lb[xj]/dom[xi]/dom[xj] + lb[xk]/dom[xi]/dom[xk]) +
                    xj*(-lb[xi]/dom[xj]/dom[xi] + lb[xk]/dom[xj]/dom[xk]) +
                    xk*(1.0/dom[xk] + lb[xi]/dom[xi]/dom[xk] + lb[xj]/dom[xj]/dom[xk])
                ) + (
                    +lb[xi]*lb[xj]/dom[xi]/dom[xj] +
                    -lb[xi]*lb[xk]/dom[xi]/dom[xk] +
                    -lb[xj]*lb[xk]/dom[xj]/dom[xk] +
                    -lb[xk]/dom[xk]
                )
                cuts.append(cut_expr >= 0)
            elif ineq_type == 3:
                cut_expr = (
                    xi_xj*(1.0/dom[xi]/dom[xj]) +
                    xj_xk*(1.0/dom[xj]/dom[xk]) +
                    xi_xk*(1.0/dom[xk]/dom[xi])
                ) + (
                    xi*(-1.0/dom[xi] - lb[xj]/dom[xi]/dom[xj] - lb[xk]/dom[xi]/dom[xk]) +
                    xj*(-1.0/dom[xj] - lb[xi]/dom[xj]/dom[xi] - lb[xk]/dom[xj]/dom[xk]) +
                    xk*(-1.0/dom[xk] - lb[xi]/dom[xi]/dom[xk] - lb[xj]/dom[xj]/dom[xk])
                ) + (
                    lb[xi]*lb[xj]/dom[xi]/dom[xj] +
                    lb[xi]*lb[xk]/dom[xi]/dom[xk] +
                    lb[xj]*lb[xk]/dom[xj]/dom[xk] +
                    lb[xi]/dom[xi] +
                    lb[xj]/dom[xj] +
                    lb[xk]/dom[xk]
                )
                cuts.append(cut_expr + 1 >= 0)
            else:
                raise RuntimeError('Invalid inequality type {}'.format(ineq_type))

        return cuts

    def _get_triangle_violations(self):
        # Evaluate violations for all valid triangle cliques and cut types
        lb = self._lower_bounds
        dom = self._domains

        clique_with_rank = self._clique_with_rank

        for idx_clique, (clique, ineq_type, _) in enumerate(clique_with_rank):
            # If the domain of any variables is very small, don't consider cut
            if any(dom[v] <= self._domain_eps for v in clique):
                continue

            x = clique[0]
            x_val = pe.value(x)
            y = clique[1]
            y_val = pe.value(y)
            z = clique[2]
            z_val = pe.value(z)

            x_y_aux = self._aux_vars[id(x), id(y)]
            x_z_aux = self._aux_vars[id(x), id(z)]
            y_z_aux = self._aux_vars[id(y), id(z)]

            x_y = (pe.value(x_y_aux) - lb[x]*y_val - lb[y]*x_val + lb[x]*lb[y])/dom[x]/dom[y]
            x_z = (pe.value(x_z_aux) - lb[x]*z_val - lb[z]*x_val + lb[x]*lb[z])/dom[x]/dom[z]
            y_z = (pe.value(y_z_aux) - lb[y]*z_val - lb[z]*y_val + lb[y]*lb[z])/dom[y]/dom[z]

            x_val_scaled = (x_val - lb[x])/dom[x]
            y_val_scaled = (y_val - lb[y])/dom[y]
            z_val_scaled = (z_val - lb[z])/dom[z]

            if ineq_type == 0:
                rank = + x_y + x_z - y_z - x_val_scaled
            elif ineq_type == 1:
                rank = + x_y - x_z + y_z - y_val_scaled
            elif ineq_type == 2:
                rank = - x_y + x_z + y_z - z_val_scaled
            elif ineq_type == 3:
                rank = - x_y - x_z - y_z + x_val_scaled + y_val_scaled + z_val_scaled - 1.0
            else:
                raise RuntimeError("Invalid clique type {}".format(ineq_type))
            self._clique_with_rank[idx_clique] = (clique, ineq_type, rank)

        return self._clique_with_rank

    def _compute_clique_ranks(self, relaxed_problem):
        lower_bounds, upper_bounds, domains, aux_vars, var_by_id, edges = \
            self._detect_bilinear_terms(relaxed_problem)

        triple_cliques = []
        for clique in enumerate_all_cliques(from_edgelist(edges)):
            if len(clique) < 3:
                continue
            elif len(clique) == 3:
                triple_cliques.append([var_by_id[i] for i in clique])
            else:
                # Cliques are sorted by length. Can safely exit.
                break

        clique_with_rank = [
            (clique, i, 0.0)
            for clique in triple_cliques
            for i in range(4)
        ]

        self._clique_with_rank = clique_with_rank
        self._lower_bounds = lower_bounds
        self._upper_bounds = upper_bounds
        self._domains = domains
        self._aux_vars = aux_vars

    def _detect_bilinear_terms(self, relaxed_problem):
        lower_bounds = pe.ComponentMap()
        upper_bounds = pe.ComponentMap()
        domains = pe.ComponentMap()

        aux_vars = dict()

        edges = []
        var_by_id = dict()

        for relaxation in relaxation_data_objects(relaxed_problem, active=True, descend_into=True):
            if isinstance(relaxation, PWMcCormickRelaxation):
                x, y = relaxation.get_rhs_vars()
                w = relaxation.get_aux_var()

                lower_bounds[w] = w.lb
                lower_bounds[x] = x.lb
                lower_bounds[y] = y.lb

                upper_bounds[w] = w.ub
                upper_bounds[x] = x.ub
                upper_bounds[y] = y.ub

                if w.has_lb() and w.has_ub():
                    domains[w] = w.ub - w.lb
                else:
                    domains[w] = np.inf

                if x.has_lb() and x.has_ub():
                    domains[x] = x.ub - x.lb
                else:
                    domains[x] = np.inf

                if y.has_lb() and y.has_ub():
                    domains[y] = y.ub - y.lb
                else:
                    domains[y] = np.inf

                aux_vars[id(x), id(y)] = aux_vars[id(y), id(x)] = w

                edges.append((id(x), id(y)))
                edges.append((id(y), id(x)))

                var_by_id[id(x)] = x
                var_by_id[id(y)] = y
            elif isinstance(relaxation, PWXSquaredRelaxation):
                x, = relaxation.get_rhs_vars()
                w = relaxation.get_aux_var()

                lower_bounds[w] = w.lb
                lower_bounds[x] = x.lb

                upper_bounds[w] = w.ub
                upper_bounds[x] = x.ub

                if w.has_lb() and w.has_ub():
                    domains[w] = w.ub - w.lb
                else:
                    domains[w] = np.inf

                if x.has_lb() and x.has_ub():
                    domains[x] = x.ub - x.lb
                else:
                    domains[x] = np.inf

                aux_vars[id(x), id(x)] = w

                edges.append((id(x), id(x)))

                var_by_id[id(x)] = x
            else:
                continue

        return lower_bounds, upper_bounds, domains, aux_vars, var_by_id, edges
