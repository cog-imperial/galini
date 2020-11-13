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
from coramin.utils.coramin_enums import RelaxationSide
from galini.branch_and_bound.algorithm import BranchAndBoundAlgorithm
from galini.branch_and_bound.node import NodeSolution
from galini.branch_and_bound.telemetry import update_counter
from galini.branch_and_cut.bound_reduction import (
    perform_obbt_on_model, perform_fbbt_on_model
)
from galini.branch_and_cut.branching import compute_branching_decision
from galini.branch_and_cut.node_storage import RootNodeStorage
from galini.branch_and_cut.extensions import (
    InitialPrimalSearchStrategyRegistry,
    PrimalHeuristicRegistry,
    NodeSelectionStrategyRegistry,
    BranchingStrategyRegistry,
    RelaxationRegistry,
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

        self._init_extensions()

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
                NumericOption(
                    'cut_violation_threshold',
                    default=1e-5,
                    description='Threshold to consider a cut violated.'
                )
            ]),
            OptionsGroup('initial_primal_search', [
                StringOption('strategy', default='default')
            ]),
            OptionsGroup('primal_heuristic', [
                StringOption('strategy', default='default')
            ]),
            OptionsGroup('branching', [
                StringOption('strategy', default='default')
            ]),
            OptionsGroup('node_selection', [
                StringOption('strategy', default='default')
            ]),
            OptionsGroup('relaxation', [
                StringOption('strategy', default='default')
            ]),
            OptionsGroup('mip_solver', [
                StringOption(
                    'name',
                    default='cplex',
                    description='MIP solver name'
                ),
                StringOption(
                    'timelimit_option',
                    default='timelimit',
                    description='The name of the option to set the MIP solver timelimit'
                ),
                StringOption(
                    'maxiter_option',
                    default='simplex limits iterations',
                    description='The name of the option to set the MIP solver maximum iterations'
                ),
                StringOption(
                    'relative_gap_option',
                    default='mip tolerances mipgap',
                    description='The name of the option to set the MIP solver relative gap tolerance'
                ),
                StringOption(
                    'absolute_gap_option',
                    default='mip tolerances absmipgap',
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

    def _init_extensions(self):
        self._init_initial_primal_search_strategy_extension()
        self._init_primal_heuristic_extension()
        self._init_branching_strategy_extension()
        self._init_node_selection_strategy_extension()
        self._init_relaxation_strategy_extension()

    def _init_initial_primal_search_strategy_extension(self):
        reg = InitialPrimalSearchStrategyRegistry()
        strategy_name = self.config['initial_primal_search']['strategy']
        strategy_cls = reg[strategy_name]
        assert strategy_cls is not None
        self._initial_primal_search_strategy = strategy_cls(self)

    def _init_primal_heuristic_extension(self):
        reg = PrimalHeuristicRegistry()
        strategy_name = self.config['primal_heuristic']['strategy']
        heuristic_cls = reg[strategy_name]
        assert heuristic_cls is not None
        self._primal_heuristic = heuristic_cls(self)

    def _init_branching_strategy_extension(self):
        reg = BranchingStrategyRegistry()
        strategy_name = self.config['branching']['strategy']
        strategy_cls = reg[strategy_name]
        assert strategy_cls is not None
        self._branching_strategy = strategy_cls(self)

    def _init_node_selection_strategy_extension(self):
        reg = NodeSelectionStrategyRegistry()
        strategy_name = self.config['node_selection']['strategy']
        strategy_cls = reg[strategy_name]
        assert strategy_cls is not None
        self._node_selection_strategy = strategy_cls(self)

    def _init_relaxation_strategy_extension(self):
        reg = RelaxationRegistry()
        strategy_name = self.config['relaxation']['strategy']
        strategy_cls = reg[strategy_name]
        assert strategy_cls is not None
        self._relaxation = strategy_cls(self)

    @property
    def branching_strategy(self):
        return self._branching_strategy

    @property
    def node_selection_strategy(self):
        return self._node_selection_strategy

    def init_node_storage(self, model):
        return RootNodeStorage(model, self._relaxation)

    @property
    def bab_config(self):
        return self._bab_config

    def _update_solver_options(self, solver, timelimit=None, relative_gap=None):
        if timelimit is None:
            # Set sub solver timelimit to something reasonable:
            #  * at least 1 second
            #  * give one second for galini to finish
            timelimit = max(self.galini.timelimit.seconds_left() - 1, 1)

        if relative_gap is None:
            relative_gap = self.bab_config['relative_gap']

        update_solver_options(
            solver,
            timelimit=timelimit,
            absolute_gap=self.bab_config['absolute_gap'],
            relative_gap=relative_gap,
        )

    def find_initial_solution(self, model, tree, node):
        try:
            return self._initial_primal_search_strategy.solve(model, tree, node)
        except Exception as ex:
            if self.galini.paranoid_mode:
                raise
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
            self.logger.info('Start FBBT {} special structure.', 'without' if is_root else 'with')
            # Skip computing mono and cvx at root node since it's recomputed after OBBT
            bounds, mono, cvx = self._perform_fbbt_on_model(tree, node, model, skip_special_structure=is_root)
        except EmptyIntervalError:
            return NodeSolution(None, None)

        with self._telemetry.timespan('branch_and_cut.model_relaxation'):
            linear_model = node.storage.model_relaxation()

        if is_root:
            obbt_time = self.bab_config['obbt_timelimit']
            if obbt_time > 0:
                self.logger.info('OBBT start')
                with self._telemetry.timespan('branch_and_cut.obbt'):
                    obbt_solver = instantiate_solver_with_options(
                        self.config['mip_solver'],
                    )

                    try:
                        new_bounds = perform_obbt_on_model(
                            obbt_solver,
                            model,
                            linear_model,
                            upper_bound=tree.upper_bound,
                            timelimit=obbt_time,
                            simplex_maxiter=self.bab_config['obbt_simplex_maxiter'],
                            absolute_gap=self.bab_config['absolute_gap'],
                            relative_gap=self.bab_config['relative_gap'],
                            mc=self.galini.mc,
                        )
                        node.storage.update_bounds(new_bounds)
                    except Exception as ex:
                        self.logger.warning('OBBT Exception: {}', ex)
                self.logger.info('OBBT completed. Starting one more round of FBBT')
                bounds, mono, cvx = self._perform_fbbt_on_model(tree, node, model)
                # Recompute linear relaxation to have better bounds on linear_model
                node.storage.recompute_model_relaxation_bounds()
                linear_model = node.storage.model_relaxation()
            else:
                self.logger.info('Skip OBBT')

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
            update_counter(self.galini, 'branch_and_cut.cut_loop_iter', cuts_state.round, 0)

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
            mip_solution = load_solution_from_model(mip_results, linear_model, solver=self._mip_solver)

        self.logger.info(
            'MILP solution after LP cut phase: {} {}',
            mip_solution.status,
            mip_solution,
        )

        if not mip_solution.status.is_success():
            # If we got this far, lower_bounding_solution is at least feasible
            return NodeSolution(lower_bounding_solution, None)

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
                model, linear_model, mip_solution, tree, node
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

    def _solve_upper_bounding_problem(self, model, linear_model, mip_solution, tree, node):
        return self._primal_heuristic.solve(model, linear_model, mip_solution, tree, node)

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
        lp_solution = None

        relaxation_data = node.storage.relaxation_data

        if node.parent:
            with self._telemetry.timespan('branch_and_cut.add_cuts_from_parent'):
                parent_cuts_count, lp_solution = self._add_cuts_from_parent(
                    node, model, linear_model
                )

        cut_loop_start_time = current_time()
        self._cut_loop_inner_iteration = 0
        first_round_required = lp_solution is None

        previous_lp_solution = None
        while first_round_required or not self.cut_loop_should_terminate(cuts_state, cut_loop_start_time):
            first_round_required = False

            feasible, new_cuts, lp_solution = self._perform_cut_round(
                model, linear_model, cuts_state, tree, node
            )

            if not feasible:
                if previous_lp_solution is not None:
                    return True, cuts_state, previous_lp_solution
                return False, cuts_state, lp_solution

            previous_lp_solution = lp_solution

            # Add cuts as constraints
            new_cuts_constraints = []
            for cut in new_cuts:
                relaxed_cut = self._relaxation.relax_inequality(linear_model, cut, RelaxationSide.BOTH, relaxation_data)
                new_cons = node.storage.cut_node_storage.add_cut(relaxed_cut)
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
                cuts_state, lp_solution
            )

            cuts_state.update(
                lp_solution,
                paranoid=self.galini.paranoid_mode,
                atol=self.bab_config['absolute_gap'],
                rtol=self.bab_config['relative_gap'],
            )

            if not new_cuts:
                break

        return True, cuts_state, lp_solution

    def _perform_cut_round(self, model, linear_model, cuts_state, tree, node):
        self.logger.debug('Round {}. Solving linearized problem.', cuts_state.round)

        self._update_solver_options(self._mip_solver)
        try:
            results = self._mip_solver.solve(linear_model, tee=True)
        except ValueError as ex:
            self.logger.warning('Error in cut round {}: {}', cuts_state.round, ex)
            return False, None, None
        mip_solution = load_solution_from_model(results, linear_model, solver=self._mip_solver)

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
            lp_solution = load_solution_from_model(results, linear_model, solver=self._mip_solver)

            if not lp_solution.status.is_success():
                break

            for cut in cut_storage.cuts:
                if cut.active:
                    continue

                if constraint_violation(cut) > self.cuts_config.cut_violation_threshold:
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
        solution = load_solution_from_model(results, model, solver=solver)
        if solution.status.is_success():
            return solution
        return None

    def _perform_fbbt_on_model(self, tree, node, model, maxiter=None, skip_special_structure=False):
        if maxiter is None:
            maxiter = self.bab_config['fbbt_maxiter']

        with self._telemetry.timespan('branch_and_cut.fbbt'):
            bounds, mono, cvx = perform_fbbt_on_model(
                model,
                tree,
                node,
                maxiter=maxiter,
                timelimit=self.bab_config['fbbt_timelimit'],
                eps=self.galini.mc.epsilon,
                skip_special_structure=skip_special_structure,
            )
            if bounds is not None:
                node.storage.update_bounds(bounds)
            return bounds, mono, cvx


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
