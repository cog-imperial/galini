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

"""Branch & Cut algorithm."""

import pyomo.environ as pe
from galini.branch_and_bound.algorithm import BranchAndBoundAlgorithm
from galini.branch_and_bound.node import NodeSolution
from galini.branch_and_bound.selection import BestLowerBoundSelectionStrategy
from galini.branch_and_cut.bound_reduction import (
    perform_obbt_on_model, perform_fbbt_on_model
)
from galini.branch_and_cut.branching import compute_branching_decision, \
    BranchAndCutBranchingStrategy
from galini.branch_and_cut.primal import (
    solve_primal, solve_primal_with_starting_point
)
from galini.branch_and_cut.state import CutsState
from galini.config import (
    OptionsGroup,
    IntegerOption,
    NumericOption,
    SolverOptions,
    StringOption,
    ExternalSolverOptions,
)
from galini.math import is_close
from galini.pyomo.util import constraint_violation, instantiate_solver_with_options, update_solver_options
from galini.solvers.solution import load_solution_from_model
from galini.timelimit import (
    current_time,
    seconds_elapsed_since,
)
from suspect.interval import EmptyIntervalError


# pylint: disable=too-many-instance-attributes
class BranchAndCutAlgorithm(BranchAndBoundAlgorithm):
    """Branch and Cut algorithm."""
    name = 'branch_and_cut'

    def __init__(self, galini):
        super().__init__(galini)
        self.galini = galini
        self._nlp_solver = instantiate_solver_with_options(
            self.config['nlp_solver'],
        )
        self._mip_solver = instantiate_solver_with_options(
            self.config['mip_solver'],
        )

        self._bab_config = self.config['bab']
        self.cuts_config = self.config['cuts']

        self._branching_strategy = BranchAndCutBranchingStrategy()
        self._node_selection_strategy = BestLowerBoundSelectionStrategy()

        # TODO(fra): refactor telemetry to support nested iterations
        self._cut_loop_outer_iteration = 0
        self._cut_loop_inner_iteration = 0

    # pylint: disable=line-too-long
    @staticmethod
    def algorithm_options():
        """Return options for BranchAndCutAlgorithm"""
        return SolverOptions(BranchAndCutAlgorithm.name, [
            OptionsGroup('bab', BranchAndBoundAlgorithm.bab_options()),
            OptionsGroup('cuts', [
                IntegerOption(
                    'maxiter',
                    default=20,
                    description='Number of cut rounds'
                ),
                NumericOption(
                    'cut_tolerance',
                    default=1e-5,
                    description='Terminate if two consecutive cut rounds are within this tolerance'
                ),
                IntegerOption(
                    'timelimit',
                    default=120,
                    description='Total timelimit for cut rounds'
                ),
            ]),
            OptionsGroup('mip_solver', [
                StringOption(
                    'name',
                    default='cplex_direct',
                    description='MIP solver name'
                ),
                StringOption(
                    'timelimit_option',
                    default='timelimit',
                    description='The name of the option to set the MIP solver timelimit'
                ),
                StringOption(
                    'maxiter_option',
                    default='simplex_limits_iterations',
                    description='The name of the option to set the MIP solver maximum iterations'
                ),
                StringOption(
                    'relative_gap_option',
                    default='mip_tolerances_mipgap',
                    description='The name of the option to set the MIP solver relative gap tolerance'
                ),
                StringOption(
                    'absolute_gap_option',
                    default='mip_tolerances_absmipgap',
                    description='The name of the option to set the MIP solver absolute gap tolerance'
                ),
                ExternalSolverOptions('options'),
            ]),
            OptionsGroup('nlp_solver', [
                StringOption(
                    'name',
                    default='ipopt',
                    description='NLP solver name'
                ),
                StringOption(
                    'timelimit_option',
                    default='max_cpu_time',
                    description='The name of the option to set the NLP solver timelimit'
                ),
                StringOption(
                    'maxiter_option',
                    default='max_iter',
                    description='The name of the option to set the NLP solver maximum iterations'
                ),
                StringOption(
                    'relative_gap_option',
                    default='tol',
                    description='The name of the option to set the NLP solver relative gap tolerance'
                ),
                StringOption(
                    'absolute_gap_option',
                    default='',
                    description='The name of the option to set the NLP solver absolute gap tolerance'
                ),
                ExternalSolverOptions('options'),
            ]),
        ])

    @property
    def branching_strategy(self):
        return self._branching_strategy

    @property
    def node_selection_strategy(self):
        return self._node_selection_strategy

    @property
    def bab_config(self):
        return self._bab_config

    def _update_solver_options(self, solver, timelimit=None):
        if timelimit is None:
            # Set sub solver timelimit to something reasonable:
            #  * at least 1 second
            #  * give one second for galini to finish
            timelimit = max(self.galini.timelimit.seconds_left() - 1, 1)

        update_solver_options(
            solver,
            timelimit=timelimit,
            absolute_gap=self.bab_config['absolute_gap'],
            relative_gap=self.bab_config['relative_gap'],
        )

    def find_initial_solution(self, model, tree, node):
        try:
            with self._telemetry.timespan('branch_and_cut.fbbt'):
                _, _, cvx = perform_fbbt_on_model(
                    model,
                    tree,
                    node,
                    maxiter=1,
                    timelimit=self.bab_config['fbbt_timelimit'],
                    eps=self.galini.mc.epsilon
                )

            solution = self._try_solve_convex_model(model, convexity=cvx)
            if solution is not None:
                return NodeSolution(solution, solution)

            # Don't pass a starting point since it's already loaded in the model
            timelimit = min(
                self.bab_config['root_node_feasible_solution_search_timelimit'],
                self.galini.timelimit.seconds_left(),
            )
            self._update_solver_options(self._nlp_solver, timelimit=timelimit)
            solution = solve_primal_with_starting_point(
                model, pe.ComponentMap(), self._nlp_solver, self.galini.mc
            )

            if solution.status.is_success():
                return NodeSolution(None, solution)
            return None
        except Exception as ex:
            if self.galini.paranoid_mode:
                raise ex
            self.logger.info('Exception in find_initial_solution: {}', ex)
            return None

    def solve_problem_at_root(self, tree, node):
        """Solve problem at root node."""
        return self._solve_problem_at_node(tree, node, True)

    def solve_problem_at_node(self, tree, node):
        """Solve problem at non root node."""
        return self._solve_problem_at_node(tree, node, False)

    def _solve_problem_at_node(self, tree, node, is_root):
        model = node.storage.model()

        try:
            with self._telemetry.timespan('branch_and_cut.fbbt'):
                bounds, mono, cvx = perform_fbbt_on_model(
                    model,
                    tree,
                    node,
                    maxiter=self.bab_config['fbbt_maxiter'],
                    timelimit=self.bab_config['fbbt_timelimit'],
                    eps=self.galini.mc.epsilon
                )
                if bounds is not None:
                    node.storage.update_bounds(bounds)
        except EmptyIntervalError:
            return NodeSolution(None, None)

        with self._telemetry.timespan('branch_and_cut.model_relaxation'):
            linear_model = node.storage.model_relaxation()

        if is_root:
            self.logger.info('OBBT start')
            with self._telemetry.timespan('branch_and_cut.obbt'):
                new_bounds = perform_obbt_on_model(
                    self._mip_solver,
                    model,
                    linear_model,
                    upper_bound=tree.upper_bound,
                    timelimit=self.bab_config['obbt_timelimit'],
                    simplex_maxiter=self.bab_config['obbt_simplex_maxiter'],
                    absolute_gap=self.bab_config['absolute_gap'],
                    relative_gap=self.bab_config['relative_gap'],
                    mc=self.galini.mc,
                )
                node.storage.update_bounds(new_bounds)
            self.logger.info('OBBT completed')

        self.logger.info(
            'Starting Cut generation iterations. Maximum iterations={}',
            self.cuts_config['maxiter'],
        )
        generators_name = [
            g.name for g in self.galini.cuts_generators_manager.generators
        ]

        self.logger.info('Using cuts generators: {}', ', '.join(generators_name))

        # Try solve the problem as convex NLP
        with self._telemetry.timespan('branch_and_cut.try_solve_convex_model'):
            solution = self._try_solve_convex_model(model, convexity=cvx)
            if solution is not None:
                return solution

        if not node.has_parent:
            assert is_root
            feasible_solution = node.initial_feasible_solution
        else:
            feasible_solution = None

        cuts_manager = self.galini.cuts_generators_manager

        if is_root:
            with self._telemetry.timespan('cuts_manager.before_start_at_root'):
                cuts_manager.before_start_at_root(model, linear_model)
        else:
            with self._telemetry.timespan('cuts_manager.before_start_at_node'):
                cuts_manager.before_start_at_node(model, linear_model)

        # Find lower bounding solution from linear model
        with self._telemetry.timespan('cuts_manager.solve_lower_bounding_relaxation'):
            feasible, cuts_state, lower_bounding_solution = self._solve_lower_bounding_relaxation(
                tree, node, model, linear_model
            )

        if is_root:
            with self._telemetry.timespan('cuts_manager.after_end_at_root'):
                cuts_manager.after_end_at_root(model, linear_model, lower_bounding_solution)
        else:
            with self._telemetry.timespan('cuts_manager.after_end_at_node'):
                cuts_manager.after_end_at_node(model, linear_model, lower_bounding_solution)

        if not feasible:
            self.logger.info('Lower bounding solution not success: {}', lower_bounding_solution)
            return NodeSolution(lower_bounding_solution, feasible_solution)

        # Check for timeout
        if self.galini.timelimit.timeout():
            return NodeSolution(lower_bounding_solution, feasible_solution)

        # Solve MILP to obtain MILP solution
        self._update_solver_options(self._mip_solver)
        with self._telemetry.timespan('branch_and_cut.solve_mip'):
            mip_results = self._mip_solver.solve(linear_model)
            mip_solution = load_solution_from_model(mip_results, linear_model)
        self.logger.info(
            'MILP solution after LP cut phase: {} {}',
            mip_solution.status,
            mip_solution,
        )

        if not self.galini.debug_assert_(
                lambda: not mip_solution.status.is_unbounded(),
                'MIP solution should not be unbounded'):
            from galini.ipython import embed_ipython
            embed_ipython(header='Unbounded MIP solution')

        if not mip_solution.status.is_success():
            return NodeSolution(mip_solution, None)

        with self._telemetry.timespan('branch_and_cut.update_node_branching_decision'):
            self._update_node_branching_decision(
                model, linear_model, mip_solution, node
            )

        assert cuts_state is not None
        can_improve_feasible_solution = not (
           cuts_state.lower_bound >= tree.upper_bound and
           not is_close(cuts_state.lower_bound, tree.upper_bound, atol=self.galini.mc.epsilon)
        )
        self.logger.debug('Can improve feasible solution? {}', can_improve_feasible_solution)
        if not can_improve_feasible_solution:
            # No improvement
            return NodeSolution(mip_solution, None)

        # Check for timeout
        if self.galini.timelimit.timeout():
            # No time for finding primal solution
            return NodeSolution(mip_solution, None)

        # Try to find a feasible solution
        with self._telemetry.timespan('branch_and_cut.solve_upper_bounding_problem'):
            primal_solution = self._solve_upper_bounding_problem(
                model, node, linear_model, mip_solution
            )

        assert primal_solution is not None, 'Should return a solution even if not feasible'

        if not primal_solution.status.is_success():
            return NodeSolution(mip_solution, feasible_solution)

        return NodeSolution(mip_solution, primal_solution)

    def _solve_lower_bounding_relaxation(self, tree, node, model, linear_model):
        self.logger.info('Solving lower bounding LP')

        originally_integer = []
        for var in linear_model.component_data_objects(pe.Var, active=True):
            if var.is_continuous():
                continue
            originally_integer.append((var, var.domain))
            var.domain = pe.Reals

        feasible, cuts_state, mip_solution = self._perform_cut_loop(
            tree, node, model, linear_model,
        )

        for var, domain in originally_integer:
            var.domain = domain

        return feasible, cuts_state, mip_solution

    def _solve_upper_bounding_problem(self, model, node, linear_model, mip_solution):
        # TODO(fra): properly map between variables
        assert mip_solution.status.is_success(), "Should be a feasible point for the relaxation"
        model_var_map = node.storage.model_to_relaxation_var_map
        mip_solution_with_model_vars = pe.ComponentMap(
            (var, mip_solution.variables[model_var_map[var]])
            for var in model.component_data_objects(pe.Var, active=True)
        )

        self._update_solver_options(self._nlp_solver)
        primal_solution = solve_primal_with_starting_point(
            model, mip_solution_with_model_vars, self._nlp_solver, self.galini.mc, fix_all=True
        )

        self._update_solver_options(self._nlp_solver)
        new_primal_solution = solve_primal(
            model, mip_solution_with_model_vars, self._nlp_solver, self.galini.mc
        )

        if new_primal_solution is not None:
            primal_solution = new_primal_solution

        return primal_solution

    def _cuts_converged(self, state):
        cuts_close = (
                state.latest_solution is not None and
                state.previous_solution is not None and
                is_close(
                    state.latest_solution,
                    state.previous_solution,
                    rtol=self.cuts_config['cut_tolerance']
                )
        )
        if cuts_close:
            return True
        return self.galini.cuts_generators_manager.has_converged(state)

    def _cuts_iterations_exceeded(self, state):
        return state.round > self.cuts_config['maxiter']

    def cut_loop_should_terminate(self, state, start_time):
        elapsed_time = seconds_elapsed_since(start_time)
        return (
            self._cuts_converged(state) or
            self._cuts_iterations_exceeded(state) or
            self.galini.timelimit.timeout() or
            elapsed_time > self.cuts_config['timelimit']
        )

    def _update_node_branching_decision(self, model, linear_model, mip_solution, node):
        weights = {
            'sum': self.bab_config['branching_weight_sum'],
            'max': self.bab_config['branching_weight_max'],
            'min': self.bab_config['branching_weight_min'],
        }
        lambda_ = self.bab_config['branching_weight_lambda']
        root_bounds = node.tree.root.storage.model_bounds
        branching_decision = compute_branching_decision(
            model, linear_model, root_bounds, mip_solution, weights, lambda_, self.galini.mc
        )
        node.storage.branching_decision = branching_decision

    def _perform_cut_loop(self, tree, node, model, linear_model):
        cuts_state = CutsState()
        mip_solution = None

        if node.parent:
            parent_cuts_count, mip_solution = self._add_cuts_from_parent(
                node, model, linear_model
            )
            # TODO(fra): output telemetry

        cut_loop_start_time = current_time()
        self._cut_loop_inner_iteration = 0
        while not self.cut_loop_should_terminate(cuts_state, cut_loop_start_time):
            feasible, new_cuts, mip_solution = self._perform_cut_round(
                model, linear_model, cuts_state, tree, node
            )

            if not feasible:
                return False, cuts_state, mip_solution

            # Add cuts as constraints
            new_cuts_constraints = []
            for cut in new_cuts:
                new_cons = node.storage.cut_node_storage.add_cut(cut)
                new_cuts_constraints.append(new_cons)

            if self.galini.paranoid_mode:
                # Check added cuts are violated
                for cons in new_cuts_constraints:
                    if not self.galini.assert_(
                            lambda: constraint_violation(cons) > 0.0,
                            'New cut must be violated'):
                        from galini.ipython import embed_ipython
                        embed_ipython(header='Cut {} must be violated'.format(cons.name))

            self.logger.debug(
                'Updating CutState: State={}, Solution={}',
                cuts_state, mip_solution
            )

            cuts_state.update(
                mip_solution,
                paranoid=self.galini.paranoid_mode,
                atol=self.bab_config['absolute_gap'],
                rtol=self.bab_config['relative_gap'],
            )

            if not new_cuts:
                break

        return True, cuts_state, mip_solution

    def _perform_cut_round(self, model, linear_model, cuts_state, tree, node):
        self.logger.debug('Round {}. Solving linearized problem.', cuts_state.round)

        self._update_solver_options(self._mip_solver)
        try:
            results = self._mip_solver.solve(linear_model)
        except ValueError as ex:
            self.logger.warning('Error in cut round {}: {}', cuts_state.round, ex)
            return False, None, None
        mip_solution = load_solution_from_model(results, model)

        self.logger.debug(
            'Round {}. Linearized problem solution is {}',
            cuts_state.round, mip_solution.status.description())
        self.logger.debug('Objective is {}'.format(mip_solution.objective))

        if not mip_solution.status.is_success():
            return False, None, mip_solution

        # Generate new cuts
        new_cuts = self.galini.cuts_generators_manager.generate(
            model, linear_model, mip_solution, tree, node
        )

        self.logger.debug(
            'Round {}. Adding {} cuts.',
            cuts_state.round, len(new_cuts)
        )

        return True, new_cuts, mip_solution

    def _add_cuts_from_parent(self, node, model, linear_model):
        self.logger.debug('Adding cuts from cut pool')
        first_loop = True
        num_violated_cuts = 0
        inherit_cuts_count = 0
        lp_solution = None

        node.storage.cut_pool.deactivate_all()

        cut_storage = node.storage.cut_node_storage

        while first_loop or num_violated_cuts > 0:
            first_loop = False
            num_violated_cuts = 0

            self._update_solver_options(self._mip_solver)
            results = self._mip_solver.solve(linear_model)
            lp_solution = load_solution_from_model(results, linear_model)

            if not lp_solution.status.is_success():
                break

            for cut in cut_storage.cuts:
                # TODO(fra): add cut violation parameter
                if cut.active:
                    continue

                if constraint_violation(cut) > 1e-5:
                    cut.activate()
                    inherit_cuts_count += 1
                    num_violated_cuts += 1

            self.logger.info(
                'Number of violated cuts at end of loop: {}',
                num_violated_cuts,
            )
        return inherit_cuts_count, lp_solution

    def _try_solve_convex_model(self, model, convexity):
        """Check if problem is continuous and convex, in that case use solve it."""
        if convexity and _is_convex(model, convexity):
            all_continuous = all(
                var.is_continuous()
                for var in model.component_data_objects(pe.Var, active=True)
            )
            if all_continuous:
                return self._solve_convex_model(model)
        return None

    def _solve_convex_model(self, model):
        self._update_solver_options(self._nlp_solver)
        solver = self._nlp_solver
        results = solver.solve(model)
        solution = load_solution_from_model(results, model)
        if solution.status.is_success():
            return solution
        return None


def _is_convex(model, cvx_map):
    is_objective_cvx = cvx_map[model._objective.expr].is_convex()

    if not is_objective_cvx:
        return False

    return all(
        _constraint_is_convex(cvx_map, cons)
        for cons in model.component_data_objects(pe.Constraint, active=True)
    )


def _constraint_is_convex(cvx_map, cons):
    cvx = cvx_map[cons.body]
    # g(x) <= UB
    if not cons.has_lb():
        return cvx.is_convex()

    # g(x) >= LB
    if not cons.has_ub():
        return cvx.is_concave()

    # LB <= g(x) <= UB
    return cvx.is_linear()
