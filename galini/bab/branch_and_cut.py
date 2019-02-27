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
"""Branch & Cut algorithm."""
from collections import namedtuple
import numpy as np
from galini.bab import BabAlgorithm, NodeSolution
from galini.abb.relaxation import AlphaBBRelaxation
from galini.core import Constraint
from galini.cuts import CutsGeneratorsManager
from galini.config import (
    OptionsGroup,
    NumericOption,
    IntegerOption,
    EnumOption,
)


class CutsState(object):
    def __init__(self):
        self.round = 0
        self.lower_bound = -np.inf
        self.first_solution = None
        self.latest_solution = None
        self.previous_solution = None

    def update(self, solution):
        self.round += 1
        current_objective = solution.objectives[0].value
        assert (current_objective >= self.lower_bound or
                np.isclose(current_objective, self.lower_bound))
        self.lower_bound = current_objective
        if self.first_solution is None:
            self.first_solution = current_objective
        else:
            self.previous_solution = self.latest_solution
            self.latest_solution = current_objective


class BranchAndCutAlgorithm(BabAlgorithm):
    def __init__(self, nlp_solver, mip_solver, cuts_generators_reg, config):
        self.initialize(config.bab)
        self._nlp_solver = nlp_solver
        self._mip_solver = mip_solver

        self._cuts_generators_manager = \
            CutsGeneratorsManager(cuts_generators_reg, config)

        bac_config = config.bab.branch_and_cut
        self.cuts_maxiter = bac_config['maxiter']
        self.cuts_relative_tolerance = bac_config['relative_tolerance']
        self.cuts_domain_eps = bac_config['domain_eps']
        self.cuts_selection_size = bac_config['selection_size']

    @staticmethod
    def algorithm_options():
        return OptionsGroup('branch_and_cut', [
            NumericOption('domain_eps',
                          default=1e-3,
                          description='Minimum domain length for each variable to consider cut on'),
            NumericOption('relative_tolerance',
                          default=1e-3,
                          description='Termination criteria on lower bound improvement between '
                                      'two consecutive cut rounds <= relative_tolerance % of '
                                      'lower bound improvement from cut round'),
            IntegerOption('maxiter', default=20, description='Number of cut rounds'),
            NumericOption('selection_size',
                          default=0.1,
                          description='Cut selection size as a % of all cuts or as absolute number of cuts'),
        ])

    def solve_problem_at_node(self, problem, tree, node):
        relaxation = AlphaBBRelaxation()
        relaxed_problem = relaxation.relax(problem)

        self.logger.info(
            'Starting Cut generation iterations. Maximum iterations={}, relative tolerance={}',
            self.cuts_maxiter,
            self.cuts_relative_tolerance)
        self.logger.info(
            'Using cuts generators: {}',
            ', '.join([g.name for g in self._cuts_generators_manager.generators]))

        cuts_state = CutsState()
        while (not self._cuts_converged(cuts_state) and
               not self._cuts_iterations_exceeded(cuts_state)):
            self.logger.info('Round {}. Solving linearized problem.', cuts_state.round)
            mip_solution = self._mip_solver.solve(relaxed_problem, logger=self.logger)
            self.logger.info(
                'Round {}. Linearized problem solution is {}',
                cuts_state.round, mip_solution.status.description())
            assert mip_solution.status.is_success()

            # Generate new cuts
            new_cuts = self._cuts_generators_manager.generate(problem, mip_solution, tree, node)
            self.logger.info('Round {}. Adding {} cuts.', cuts_state.round, len(new_cuts))

            # Add cuts as constraints
            # TODO(fra): use problem global and local cuts
            for cut in new_cuts:
                new_cons = Constraint(cut.name, cut.expr, cut.lower_bound, cut.upper_bound)
                relaxation._relax_constraint(problem, relaxed_problem, new_cons)

            cuts_state.update(mip_solution)

        if cuts_state.lower_bound >= tree.upper_bound:
            # No improvement
            return NodeSolution(
                cuts_state.lower_bound,
                np.inf,
                mip_solution,
            )

        # Solve original problem
        nlp_solution = self._nlp_solver.solve(problem, logger=self.logger)
        if nlp_solution.status.is_success():
            upper_bound = nlp_solution.objectives[0].value
        else:
            upper_bound = np.inf
        return NodeSolution(
            cuts_state.lower_bound,
            upper_bound,
            nlp_solution,
        )

    def _solve_problem_at_root(self, problem, tree, node):
        self._perform_fbbt(problem)
        self._cuts_generators_manager.before_start_at_root(problem)
        solution = self.solve_problem_at_node(problem, tree, node)
        self._cuts_generators_manager.after_end_at_root(problem, solution)
        return solution

    def _solve_problem_at_node(self, problem, tree, node):
        self._perform_fbbt(problem)
        self._cuts_generators_manager.before_start_at_node(problem)
        solution = self.solve_problem_at_node(problem, tree, node)
        self._cuts_generators_manager.after_end_at_node(problem, solution)
        return solution

    def _cuts_converged(self, state):
        """Termination criteria for cut generation loop.

        Termination criteria on lower bound improvement between two consecutive
        cut rounds <= relative_tolerance % of lower bound improvement from cut round.
        """
        if (state.first_solution is None or
            state.previous_solution is None or
            state.latest_solution is None):
            return False

        if np.isclose(state.latest_solution, state.previous_solution):
            return True

        improvement = state.latest_solution - state.previous_solution
        lower_bound_improvement = state.latest_solution - state.first_solution
        return (improvement / lower_bound_improvement) <= state.cuts_relative_tolerance

    def _cuts_iterations_exceeded(self, state):
        return state.round > self.cuts_maxiter
