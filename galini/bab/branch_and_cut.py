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
from suspect.expression import ExpressionType
from galini.bab import BabAlgorithm, NodeSolution
from galini.abb.relaxation import AlphaBBRelaxation
from galini.logging import get_logger
from galini.core import Constraint, LinearExpression, Domain, Sense
from galini.cuts import CutsGeneratorsManager
from galini.util import print_problem
from galini.config import (
    OptionsGroup,
    NumericOption,
    IntegerOption,
    EnumOption,
)


logger = get_logger(__name__)


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
    name = 'branch_and_cut'

    def __init__(self, galini):
        self.galini = galini
        self._nlp_solver = galini.instantiate_solver('ipopt')
        self._mip_solver = galini.instantiate_solver('mip')
        self._cuts_generators_manager = galini.cuts_generators_manager
        self.initialize(galini.get_configuration_group('bab'))

        bac_config = galini.get_configuration_group('bab.branch_and_cut')
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

    def solve_problem_at_node(self, run_id, problem, relaxed_problem, tree, node, relaxation):
        logger.info(
            run_id,
            'Starting Cut generation iterations. Maximum iterations={}, relative tolerance={}',
            self.cuts_maxiter,
            self.cuts_relative_tolerance)
        logger.info(
            run_id,
            'Using cuts generators: {}',
            ', '.join([g.name for g in self._cuts_generators_manager.generators]))

        linear_problem = self._linear_problem(relaxed_problem)
        cuts_state = CutsState()
        while (not self._cuts_converged(cuts_state) and
               not self._cuts_iterations_exceeded(cuts_state)):
            new_cuts, mip_solution = \
                self._perform_cut_round(run_id, problem, relaxed_problem, linear_problem, cuts_state, tree, node)

            # Add cuts as constraints
            # TODO(fra): use problem global and local cuts
            for cut in new_cuts:
                new_cons = Constraint(cut.name, cut.expr, cut.lower_bound, cut.upper_bound)
                added_cons = relaxation._relax_constraint(problem, relaxed_problem, new_cons)
                linear_problem.add_constraint(
                    new_cons.name,
                    _convert_linear_expr(linear_problem, added_cons.root_expr),
                    new_cons.lower_bound,
                    new_cons.upper_bound,
                )

            cuts_state.update(mip_solution)
            if len(new_cuts) == 0:
                break

        logger.info(
            run_id,
            'Lower Bound from MIP = {}; Tree Upper Bound = {}',
            cuts_state.lower_bound,
            tree.upper_bound
        )

        if cuts_state.lower_bound >= tree.upper_bound and not np.isclose(cuts_state.lower_bound, tree.upper_bound):
            # No improvement
            return NodeSolution(
                cuts_state.lower_bound,
                np.inf,
                mip_solution,
            )

        primal_solution = self._solve_primal(problem, mip_solution)

        if primal_solution.status.is_success() or primal_solution.status.is_iterations_exceeded():
            upper_bound = primal_solution.objectives[0].value
        else:
            upper_bound = np.inf

        return NodeSolution(
            cuts_state.lower_bound,
            upper_bound,
            primal_solution,
        )

    def _solve_problem_at_root(self, run_id, problem, tree, node):
        self._perform_fbbt(run_id, problem, tree, node)
        relaxation, relaxed_problem = self._relax_problem(problem)
        self._cuts_generators_manager.before_start_at_root(run_id, problem, relaxed_problem)
        solution = self.solve_problem_at_root(run_id, problem, relaxed_problem, tree, node, relaxation)
        self._cuts_generators_manager.after_end_at_root(run_id, problem, relaxed_problem, solution)
        return solution

    def _solve_problem_at_node(self, run_id, problem, tree, node):
        self._perform_fbbt(run_id, problem, tree, node)
        relaxation, relaxed_problem = self._relax_problem(problem)
        self._cuts_generators_manager.before_start_at_node(run_id, problem, relaxed_problem)
        solution = self.solve_problem_at_node(run_id, problem, relaxed_problem, tree, node, relaxation)
        self._cuts_generators_manager.after_end_at_node(run_id, problem, relaxed_problem, solution)
        return solution

    def _relax_problem(self, problem):
        relaxation = AlphaBBRelaxation()
        relaxed_problem = relaxation.relax(problem)
        return relaxation, relaxed_problem

    def _linear_problem(self, problem):
        linear = problem.make_relaxed(problem.name + '_linear')
        # Add a variable that acts as objective value
        objvar = linear.add_variable('_objvar', None, None, Domain.REAL)
        linear.add_objective('_objective', LinearExpression([objvar], [1.0], 0.0), Sense.MINIMIZE)
        # add linear objective and constraints
        for objective in problem.objectives:
            root_expr = objective.root_expr
            if root_expr.expression_type == ExpressionType.Linear:
                new_root_expr = _convert_linear_expr(linear, root_expr)
                children = [c for c in new_root_expr.children]
                children.append(objvar)
                coefficients = [new_root_expr.coefficient(c) for c in new_root_expr.children]
                coefficients.append(-1.0)
                final_root_expr = LinearExpression(children, coefficients, new_root_expr.constant_term)
                linear.add_constraint(objective.name, final_root_expr, None, 0)
            elif root_expr.expression_type == ExpressionType.Sum:
                if not all([ch.expression_type == ExpressionType.Linear for ch in root_expr.children]):
                    continue
                new_root_expr = _convert_linear_expr(linear, root_expr)
                children = [c for c in new_root_expr.children]
                children.append(objvar)
                coefficients = [new_root_expr.coefficient(c) for c in new_root_expr.children]
                coefficients.append(-1.0)
                final_root_expr = LinearExpression(children, coefficients, new_root_expr.constant_term)
                linear.add_constraint(objective.name, final_root_expr, None, 0)
            elif root_expr.expression_type == ExpressionType.Variable:
                new_root_expr = LinearExpression([root_expr, objvar], [1.0, -1.0], 0.0)
                linear.add_constraint(objective.name, new_root_expr, None, 0)
            else:
                raise ValueError('Unknown expression {} of type {}'.format(root_expr, root_expr.expression_type))

        for constraint in problem.constraints:
            root_expr = constraint.root_expr
            if root_expr.expression_type == ExpressionType.Linear:
                new_root_expr = _convert_linear_expr(linear, root_expr)
                linear.add_constraint(constraint.name, new_root_expr, constraint.lower_bound, constraint.upper_bound)
            elif root_expr.expression_type == ExpressionType.Sum:
                if not all([ch.expression_type == ExpressionType.Linear for ch in root_expr.children]):
                    continue
                new_root_expr = _convert_linear_expr(linear, root_expr)
                linear.add_constraint(constraint.name, new_root_expr, constraint.lower_bound, constraint.upper_bound)
            elif root_expr.expression_type == ExpressionType.Variable:
                new_root_expr = LinearExpression([root_expr], [1.0], 0.0)
                linear.add_constraint(constraint.name, new_root_expr, constraint.lower_bound, constraint.upper_bound)
            else:
                raise ValueError('Unknown expression {} of type {}'.format(root_expr, root_expr.expression_type))
        return linear

    def _perform_cut_round(self, run_id, problem, relaxed_problem, linear_problem, cuts_state, tree, node):
        logger.info(run_id, 'Round {}. Solving linearized problem.', cuts_state.round)

        mip_solution = self._mip_solver.solve(linear_problem)

        logger.info(
            run_id,
            'Round {}. Linearized problem solution is {}',
            cuts_state.round, mip_solution.status.description())
        logger.debug(run_id, 'Objective is {}'.format(mip_solution.objectives))
        logger.debug(run_id, 'Variables are {}'.format(mip_solution.variables))
        assert mip_solution.status.is_success()

        # Generate new cuts
        new_cuts = self._cuts_generators_manager.generate(
            run_id, problem, relaxed_problem, mip_solution, tree, node)
        logger.info(run_id, 'Round {}. Adding {} cuts.', cuts_state.round, len(new_cuts))
        return new_cuts, mip_solution

    def _solve_primal(self, problem, mip_solution):
        # Solve original problem
        # Use mip solution as starting point
        for v, sv in zip(problem.variables, mip_solution.variables):
            domain = problem.domain(v)
            view = problem.variable_view(v)
            if sv.value is None:
                lb = view.lower_bound()
                if lb is None:
                    lb = -2e19
                ub = view.upper_bound()
                if ub is None:
                    ub = 2e19

                value = lb + (ub - lb) / 2.0
            else:
                value = sv.value
            if domain != Domain.REAL:
                problem.fix(v, value)
            else:
                problem.set_starting_point(v, value)

        solution = self._nlp_solver.solve(problem)

        # unfix all variables
        for v in problem.variables:
            problem.unfix(v)
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
        return (improvement / lower_bound_improvement) <= self.cuts_relative_tolerance

    def _cuts_iterations_exceeded(self, state):
        return state.round > self.cuts_maxiter


def _convert_linear_expr(linear_problem, expr):
    stack = [expr]
    coefficients = {}
    const = 0.0
    while len(stack) > 0:
        expr = stack.pop()
        if expr.expression_type == ExpressionType.Sum:
            for ch in expr.children:
                stack.append(ch)
        elif expr.expression_type == ExpressionType.Linear:
            const += expr.constant_term
            for ch in expr.children:
                if ch.idx not in coefficients:
                    coefficients[ch.idx] = 0
                coefficients[ch.idx] += expr.coefficient(ch)
        else:
            raise ValueError('Invalid ExpressionType {}'.format(expr.expression_type))

    children = []
    coeffs = []
    for var, coef in coefficients.items():
        children.append(linear_problem.variable(var))
        coeffs.append(coef)
    return LinearExpression(children, coeffs, const)
