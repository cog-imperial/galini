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

import numpy as np
from suspect.expression import ExpressionType
from suspect.interval import Interval

import galini.core as core
from galini.branch_and_bound.node import NodeSolution
from galini.branch_and_bound.relaxations import (
    ConvexRelaxation, LinearRelaxation
)
from galini.branch_and_bound.selection import BestLowerBoundSelectionStrategy
from galini.branch_and_bound.strategy import KSectionBranchingStrategy
from galini.branch_and_cut.bound_reduction import (
    perform_obbt_on_model, best_upper_bound, best_lower_bound
)
from galini.branch_and_cut.primal import (
    solve_primal, solve_primal_with_starting_point
)
from galini.branch_and_cut.state import CutsState
from galini.config import (
    OptionsGroup,
    IntegerOption,
    BoolOption,
    NumericOption,
)
from galini.cuts.generator import Cut
from galini.logging import get_logger, DEBUG
from galini.math import mc, is_close, almost_ge, almost_le, is_inf
from galini.quantities import relative_gap, absolute_gap
from galini.relaxations.relaxed_problem import RelaxedProblem
from galini.special_structure import (
    propagate_special_structure,
    perform_fbbt,
)
from galini.timelimit import (
    seconds_left,
    current_time,
    seconds_elapsed_since,
)
from galini.util import expr_to_str

logger = get_logger(__name__)


# pylint: disable=too-many-instance-attributes
class BranchAndCutAlgorithm:
    """Branch and Cut algorithm."""
    name = 'branch_and_cut'

    def __init__(self, galini, solver, telemetry):
        self.galini = galini
        self.solver = solver
        self._nlp_solver = galini.instantiate_solver('ipopt')
        self._mip_solver = galini.instantiate_solver('mip')
        self._cuts_generators_manager = galini.cuts_generators_manager
        self._bac_telemetry = telemetry

        bab_config = galini.get_configuration_group(solver.name)

        self.tolerance = bab_config['tolerance']
        self.relative_tolerance = bab_config['relative_tolerance']
        self.node_limit = bab_config['node_limit']
        self.fbbt_maxiter = bab_config['fbbt_maxiter']
        self.fbbt_timelimit = bab_config['fbbt_timelimit']
        self.root_node_feasible_solution_seed = \
            bab_config['root_node_feasible_solution_seed']

        self.root_node_feasible_solution_search_timelimit = \
            bab_config['root_node_feasible_solution_search_timelimit']

        bac_config = galini.get_configuration_group('branch_and_cut.cuts')
        self.cuts_maxiter = bac_config['maxiter']
        self._cuts_timelimit = bac_config['timelimit']
        self._cut_tolerance = bac_config['cut_tolerance']
        self._use_lp_cut_phase = bac_config['use_lp_cut_phase']
        self._use_milp_cut_phase = bac_config['use_milp_cut_phase']

        if not self._use_lp_cut_phase and not self._use_milp_cut_phase:
            raise RuntimeError('One of LP or MILP cut phase must be active')

        self.branching_strategy = KSectionBranchingStrategy(2)
        self.node_selection_strategy = BestLowerBoundSelectionStrategy()

        self._user_model = None
        self._bounds = None
        self._monotonicity = None
        self._convexity = None

        # TODO(fra): refactor telemetry to support nested iterations
        self._cut_loop_outer_iteration = 0
        self._cut_loop_inner_iteration = 0

    # pylint: disable=line-too-long
    @staticmethod
    def algorithm_options():
        """Return options for BranchAndCutAlgorithm"""
        return OptionsGroup('cuts', [
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
            BoolOption(
                'use_lp_cut_phase',
                default=True,
                description='Solve LP in cut loop'
            ),
            BoolOption(
                'use_milp_cut_phase',
                default=False,
                description='Add additional cut loop solving MILP'
            ),
        ])

    def _before_root_node(self, problem, upper_bound):
        if self._user_model is None:
            raise RuntimeError("No user model. Did you call 'before_solve'?")
        obbt_upper_bound = None
        if upper_bound is not None and not is_inf(upper_bound):
            obbt_upper_bound = upper_bound

        model = self._user_model
        obbt_start_time = current_time()
        try:
            perform_obbt_on_model(
                model, problem, obbt_upper_bound,
                timelimit=self.solver.config['obbt_timelimit'],
                simplex_maxiter=self.solver.config['obbt_simplex_maxiter'],
            )
            self._bac_telemetry.increment_obbt_time(
                seconds_elapsed_since(obbt_start_time)
            )
        except TimeoutError:
            logger.info(0, 'OBBT timed out')
            self._bac_telemetry.increment_obbt_time(
                seconds_elapsed_since(obbt_start_time)
            )
            return

        except Exception as ex:
            logger.warning(0, 'Error performing OBBT: {}', ex)
            self._bac_telemetry.increment_obbt_time(
                seconds_elapsed_since(obbt_start_time)
            )
            raise

    def before_solve(self, model, problem):
        """Callback called before solve."""
        self._user_model = model

    def _has_converged(self, state):
        rel_gap = relative_gap(state.lower_bound, state.upper_bound)
        abs_gap = absolute_gap(state.lower_bound, state.upper_bound)

        bounds_close = is_close(
            state.lower_bound,
            state.upper_bound,
            rtol=self.relative_tolerance,
            atol=self.tolerance,
        )

        if self.galini.paranoid_mode:
            assert (state.lower_bound <= state.upper_bound or bounds_close)

        return (
            rel_gap <= self.relative_tolerance or
            abs_gap <= self.tolerance
        )

    def _cuts_converged(self, state):
        cuts_close =  (
                state.latest_solution is not None and
                state.previous_solution is not None and
                is_close(
                    state.latest_solution,
                    state.previous_solution,
                    rtol=self._cut_tolerance
                )
        )
        if cuts_close:
            return True
        return self._cuts_generators_manager.has_converged(state)

    def _cuts_iterations_exceeded(self, state):
        return state.round > self.cuts_maxiter

    def cut_loop_should_terminate(self, state, start_time):
        elapsed_time = seconds_elapsed_since(start_time)
        return (
            self._cuts_converged(state) or
            self._cuts_iterations_exceeded(state) or
            self._timeout() or
            elapsed_time > self._cuts_timelimit
        )

    def _node_limit_exceeded(self, state):
        return state.nodes_visited > self.node_limit

    def _timeout(self):
        return seconds_left() <= 0

    def should_terminate(self, state):
        return (
            self._has_converged(state) or
            self._node_limit_exceeded(state) or
            self._timeout()
        )

    # pylint: disable=too-many-branches
    def _solve_problem_at_node(self, run_id, problem, relaxed_problem,
                               tree, node):
        logger.info(
            run_id,
            'Starting Cut generation iterations. Maximum iterations={}',
            self.cuts_maxiter)
        generators_name = [
            g.name for g in self._cuts_generators_manager.generators
        ]
        logger.info(
            run_id,
            'Using cuts generators: {}',
            ', '.join(generators_name)
        )

        logger.debug(run_id, 'Variables bounds of problem')
        for v in problem.variables:
            vv = problem.variable_view(v)
            logger.debug(run_id, '\t{}\t({}, {})', v.name,
                         vv.lower_bound(), vv.upper_bound())

        solution = self._try_solve_convex_problem(problem)
        if solution is not None:
            return solution

        if not node.has_parent:
            feasible_solution = node.initial_feasible_solution
        else:
            feasible_solution = None

        if logger.level <= DEBUG:
            logger.debug(run_id, 'Relaxed Problem')
            logger.debug(run_id, 'Variables:')
            relaxed = relaxed_problem

            for v in relaxed.variables:
                vv = relaxed.variable_view(v)
                logger.debug(
                    run_id, '\t{}: [{}, {}] c {}',
                    v.name, vv.lower_bound(), vv.upper_bound(), vv.domain
                )
            logger.debug(
                run_id, 'Objective: {}',
                expr_to_str(relaxed.objective.root_expr)
            )
            logger.debug(run_id, 'Constraints:')
            for constraint in relaxed.constraints:
                logger.debug(
                    run_id,
                    '{}: {} <= {} <= {}',
                    constraint.name,
                    constraint.lower_bound,
                    expr_to_str(constraint.root_expr),
                    constraint.upper_bound,
                )

        linear_problem = self._build_linear_relaxation(relaxed_problem)

        cuts_state = None
        lower_bound_search_start_time = current_time()
        if self._use_lp_cut_phase:
            logger.info(run_id, 'Start LP cut phase')
            originally_integer = []
            if not self._use_milp_cut_phase:
                for var in linear_problem.relaxed.variables:
                    vv = linear_problem.relaxed.variable_view(var)
                    if vv.domain.is_integer():
                        originally_integer.append(var)
                        linear_problem.relaxed.set_domain(var, core.Domain.REAL)

            feasible, cuts_state, mip_solution = self._perform_cut_loop(
                run_id, tree, node, problem, relaxed_problem, linear_problem,
            )

            for var in originally_integer:
                linear_problem.relaxed.set_domain(var, core.Domain.INTEGER)

            if not feasible:
                logger.info(run_id, 'LP solution is not feasible')
                self._bac_telemetry.increment_lower_bound_time(
                    seconds_elapsed_since(lower_bound_search_start_time)
                )
                return NodeSolution(mip_solution, feasible_solution)

            # Solve MILP to obtain MILP solution
            mip_solution = self._mip_solver.solve(linear_problem.relaxed)
            logger.info(
                run_id,
                'MILP solution after LP cut phase: {} {}',
                mip_solution.status,
                mip_solution,
            )
            if mip_solution.status.is_success():
                logger.update_variable(
                    run_id,
                    iteration=self._cut_loop_outer_iteration,
                    var_name='milp_solution',
                    value=mip_solution.objective_value()
                )

        xx_s = dict()
        xx_max = dict()
        xx_min = dict()

        unbounded_vars = []
        for var in problem.variables:
            if is_inf(problem.lower_bound(var)) and is_inf(problem.upper_bound(var)):
                unbounded_vars.append(var.idx)

        for var in linear_problem.relaxed.variables:
            #print('o ', var.name, mip_solution.variables[var.idx], var.reference)
            if not mip_solution.status.is_success():
                continue
            if var.reference:
                if not hasattr(var.reference, 'var1'):
                    continue
                v1 = var.reference.var1
                v2 = var.reference.var2
                #print(var.name, var.reference.var1.name, var.reference.var2.name)
                w_xk = mip_solution.variables[var.idx].value
                v1_xk = mip_solution.variables[v1.idx].value
                v2_xk = mip_solution.variables[v2.idx].value

                if v1_xk is None or v2_xk is None:
                    continue

                err = np.abs(w_xk - v1_xk*v2_xk) / (1 + np.sqrt(v1_xk**2.0 + v2_xk**2.0))

                if v1.idx not in xx_s:
                    xx_s[v1.idx] = 0.0
                    xx_max[v1.idx] = -np.inf
                    xx_min[v1.idx] = np.inf
                if v2.idx not in xx_s:
                    xx_s[v2.idx] = 0.0
                    xx_max[v2.idx] = -np.inf
                    xx_min[v2.idx] = np.inf
                xx_s[v1.idx] += err
                xx_s[v2.idx] += err
                xx_max[v1.idx] = max(xx_max[v1.idx], err)
                xx_max[v2.idx] = max(xx_max[v2.idx], err)
                xx_min[v1.idx] = min(xx_min[v1.idx], err)
                xx_min[v2.idx] = min(xx_min[v2.idx], err)
                #print()

        m_v = None
        for v in xx_s.keys():
            vv = problem.variable_view(v)
            if m_v is None:
                if is_close(vv.lower_bound(), vv.upper_bound(), atol=1e-5):
                    continue
                m_v = v
            else:
                if xx_s[v] > xx_s[m_v]:
                    if is_close(vv.lower_bound(), vv.upper_bound(), atol=1e-5):
                        continue
                    m_v = v

        if len(unbounded_vars) > 0:
            m_v = unbounded_vars[0]
            node.storage._branching_var = problem.variable_view(m_v)
            node.storage._branching_point = 0.0
        elif m_v is not None:
            node.storage._branching_var = problem.variable_view(m_v)
            vv = problem.variable_view(m_v)
            #point = 0.25 * (vv.lower_bound() + 0.5 * (vv.upper_bound() - vv.lower_bound())) + 0.75 * mip_solution.variables[m_v].value
            #point = mip_solution.variables[m_v].value
            lb = vv.lower_bound()
            ub = vv.upper_bound()
            if is_inf(lb):
                lb = -mc.user_upper_bound
            if is_inf(ub):
                ub = mc.user_upper_bound
            lambda_ = 0.25
            point = lambda_ * (lb + 0.5 * (ub - lb)) + (1 - lambda_) * mip_solution.variables[m_v].value
            node.storage._branching_point = point
            print('Branching on ', node.coordinate, node.storage._branching_var.name, point)
        else:
            node.storage._branching_variable = None
            node.storage._branching_point = None
        #input('BBB Continue... ')

        if self._use_milp_cut_phase:
            logger.info(run_id, 'Using MILP cut phase')
            feasible, cuts_state, mip_solution = self._perform_cut_loop(
                run_id, tree, node, problem, relaxed_problem, linear_problem,
            )

            if not feasible:
                logger.info(run_id, 'MILP cut phase solution is not feasible')
                self._bac_telemetry.increment_lower_bound_time(
                    seconds_elapsed_since(lower_bound_search_start_time)
                )
                return NodeSolution(mip_solution, feasible_solution)

        assert cuts_state is not None
        self._bac_telemetry.increment_lower_bound_time(
            seconds_elapsed_since(lower_bound_search_start_time)
        )

        if cuts_state.lower_bound >= tree.upper_bound and \
                not is_close(cuts_state.lower_bound, tree.upper_bound,
                             atol=mc.epsilon):
            # No improvement
            return NodeSolution(mip_solution, None)

        if self._timeout():
            # No time for finding primal solution
            return NodeSolution(mip_solution, None)

        upper_bound_search_start_time = current_time()

        starting_point = [v.value for v in mip_solution.variables]
        primal_solution = solve_primal_with_starting_point(
            run_id, problem, starting_point, self._nlp_solver, fix_all=True
        )
        new_primal_solution = solve_primal(
            run_id, problem, mip_solution, self._nlp_solver
        )
        if new_primal_solution is not None:
            primal_solution = new_primal_solution

        self._bac_telemetry.increment_upper_bound_time(
            seconds_elapsed_since(upper_bound_search_start_time)
        )

        if not primal_solution.status.is_success() and \
                feasible_solution is not None:
            # Could not get primal solution, but have a feasible solution
            return NodeSolution(mip_solution, feasible_solution)

        return NodeSolution(mip_solution, primal_solution)

    def _perform_cut_loop(self, run_id, tree, node, problem, relaxed_problem,
                          linear_problem):
        cuts_state = CutsState()
        mip_solution = None

        if node.parent:
            parent_cuts_count, mip_solution = self._add_cuts_from_parent(
                run_id, node, relaxed_problem, linear_problem
            )
            if self._bac_telemetry:
                self._bac_telemetry.increment_inherited_cuts(parent_cuts_count)

        cut_loop_start_time = current_time()
        self._cut_loop_inner_iteration = 0
        while not self.cut_loop_should_terminate(cuts_state, cut_loop_start_time):
            feasible, new_cuts, mip_solution = self._perform_cut_round(
                run_id, problem, relaxed_problem,
                linear_problem.relaxed, cuts_state, tree, node
            )

            if not feasible:
                return False, cuts_state, mip_solution

            # output
            logger.update_variable(
                run_id,
                iteration=[
                    self._cut_loop_outer_iteration,
                    self._cut_loop_inner_iteration
                ],
                var_name='cut_loop_lower_bound',
                value=mip_solution.objective_value()
            )
            self._cut_loop_inner_iteration += 1

            # Add cuts as constraints
            new_cuts_constraints = []
            for cut in new_cuts:
                node.storage.cut_pool.add_cut(cut)
                node.storage.cut_node_storage.add_cut(cut)
                new_cons = _add_cut_to_problem(linear_problem, cut)
                new_cuts_constraints.append(new_cons)

            if self.galini.paranoid_mode:
                # Check added constraints are violated
                linear_problem_tree_data = \
                    linear_problem.relaxed.expression_tree_data()
                mip_x = [x.value for x in mip_solution.variables]
                linear_problem_eval = linear_problem_tree_data.eval(
                    mip_x,
                    [new_cons.root_expr.idx for new_cons in new_cuts_constraints]
                )
                linear_problem_x = linear_problem_eval.forward(0, mip_x)
                for i, cons in enumerate(new_cuts_constraints):
                    if cons.lower_bound is not None:
                        assert cons.lower_bound >= linear_problem_x[i]
                    if cons.upper_bound is not None:
                        assert cons.upper_bound <= linear_problem_x[i]

            logger.debug(
                run_id, 'Updating CutState: State={}, Solution={}',
                cuts_state, mip_solution
            )

            cuts_state.update(
                mip_solution,
                paranoid=self.galini.paranoid_mode,
                atol=self.tolerance,
                rtol=self.relative_tolerance,
            )

            self._bac_telemetry.increment_total_cut_rounds()

            if not new_cuts:
                break

        return True, cuts_state, mip_solution

    def find_initial_solution(self, run_id, problem, tree, node):
        def _get_starting_point(v):
            vv = problem.variable_view(v)
            # Use starting point if present
            if vv.has_starting_point():
                return vv.starting_point()
            # If var has both bounds, use midpoint
            if vv.lower_bound() is not None and vv.upper_bound is not None:
                return (
                        vv.lower_bound() +
                        0.5 * (vv.upper_bound() - vv.lower_bound())
                )
            # If unbounded, use 0
            if vv.lower_bound() is None and vv.upper_bound() is None:
                return 0.0
            # If no lower bound, use upper bound
            if vv.lower_bound() is None:
                return vv.upper_bound()
            # Otherwise, use lower bound
            return vv.lower_bound()

        try:
            self._perform_fbbt(run_id, problem, tree, node, maxiter=1)
            solution = self._try_solve_convex_problem(problem)
            if solution is not None:
                return solution
            # Pass the user-given starting point to the primal heuristic,
            # then solve the problem to find an initial feasible solution.
            relaxed_problem = self._build_convex_relaxation(problem)
            starting_point = [
                _get_starting_point(v)
                for v in problem.variables
            ]

            solution = solve_primal_with_starting_point(
                run_id, problem, starting_point, self._nlp_solver
            )
            return NodeSolution(None, solution)
        except Exception as ex:
            if self.galini.paranoid_mode:
                raise ex
            logger.info(run_id, 'Exception in find_initial_solution: {}', ex)
            return None

    def solve_problem_at_root(self, run_id, problem, tree, node):
        """Solve problem at root node."""
        self._before_root_node(problem, tree.upper_bound)
        self._perform_fbbt(run_id, problem, tree, node)
        relaxed_problem = self._build_convex_relaxation(problem)
        self._cuts_generators_manager.before_start_at_root(
            run_id, problem, relaxed_problem.relaxed
        )

        solution = self._solve_problem_at_node(
            run_id, problem, relaxed_problem.relaxed, tree, node
        )
        self._cuts_generators_manager.after_end_at_root(
            run_id, problem, relaxed_problem.relaxed, solution
        )
        self._bounds = None
        self._convexity = None
        self._monotonicity = None
        self._cut_loop_outer_iteration += 1
        return solution

    def solve_problem_at_node(self, run_id, problem, tree, node):
        """Solve problem at non root node."""
        self._perform_fbbt(run_id, problem, tree, node)
        relaxed_problem = self._build_convex_relaxation(problem)

        self._cuts_generators_manager.before_start_at_node(
            run_id, problem, relaxed_problem.relaxed
        )
        solution = self._solve_problem_at_node(
            run_id, problem, relaxed_problem.relaxed, tree, node
        )
        self._cuts_generators_manager.after_end_at_node(
            run_id, problem, relaxed_problem.relaxed, solution
        )
        self._bounds = None
        self._convexity = None
        self._monotonicity = None
        self._cut_loop_outer_iteration += 1
        return solution

    def _build_convex_relaxation(self, problem):
        relaxation = ConvexRelaxation(
            problem,
            self._bounds,
            self._monotonicity,
            self._convexity,
        )
        return RelaxedProblem(relaxation, problem)

    def _build_linear_relaxation(self, problem):
        relaxation = LinearRelaxation(
            problem,
            self._bounds,
            self._monotonicity,
            self._convexity,
        )
        return RelaxedProblem(relaxation, problem)

    def _perform_cut_round(self, run_id, problem, relaxed_problem,
                           linear_problem, cuts_state, tree, node):
        logger.debug(run_id, 'Round {}. Solving linearized problem.', cuts_state.round)

        mip_solution = self._mip_solver.solve(linear_problem)

        logger.debug(
            run_id,
            'Round {}. Linearized problem solution is {}',
            cuts_state.round, mip_solution.status.description())
        logger.debug(run_id, 'Objective is {}'.format(mip_solution.objective))
        logger.debug(run_id, 'Variables are {}'.format(mip_solution.variables))
        if not mip_solution.status.is_success():
            return False, None, mip_solution
        # assert mip_solution.status.is_success()

        # Generate new cuts
        new_cuts = self._cuts_generators_manager.generate(
            run_id, problem, relaxed_problem, linear_problem, mip_solution,
            tree, node
        )
        logger.debug(
            run_id, 'Round {}. Adding {} cuts.',
            cuts_state.round, len(new_cuts)
        )
        return True, new_cuts, mip_solution

    def _find_root_node_feasible_solution(self, run_id, problem):
        logger.info(run_id, 'Finding feasible solution at root node')

        if self.root_node_feasible_solution_seed is not None:
            seed = self.root_node_feasible_solution_seed
            logger.info(run_id, 'Use numpy seed {}', seed)
            np.random.seed(seed)

        starting_point = [0.0] * problem.num_variables
        for i, v in enumerate(problem.variables):
            if problem.has_starting_point(v):
                starting_point[i] = problem.starting_point(v)
            elif problem.domain(v).is_integer():
                # Variable is integer and will be fixed, but we don't have a
                # starting point for it. Use lower or upper bound.
                has_lb = is_inf(problem.lower_bound(v))
                has_ub = is_inf(problem.upper_bound(v))
                if has_lb:
                    starting_point[i] = problem.lower_bound(v)
                elif has_ub:
                    starting_point[i] = problem.upper_bound(v)
                else:
                    starting_point[i] = 0
        return solve_primal_with_starting_point(
            run_id, problem, starting_point, self._nlp_solver
        )

    def _add_cuts_from_parent(self, run_id, node, relaxed_problem, linear_problem):
        logger.info(run_id, 'Adding inherit cuts.')
        first_loop = True
        num_violated_cuts = 0
        inherit_cuts_count = 0
        lp_solution = None
        while first_loop or num_violated_cuts > 0:
            first_loop = False
            num_violated_cuts = 0

            lp_solution = self._mip_solver.solve(linear_problem.relaxed)
            variables_x = [
                v.value
                for v in lp_solution.variables[:relaxed_problem.num_variables]
            ]

            # If the LP does not contain a variable, it's solution will be
            # None. Fix to (valid) numerical value so that we can evaluate
            # the expression.
            for i, var in enumerate(relaxed_problem.variables):
                if variables_x[i] is None:
                    lb = relaxed_problem.lower_bound(var)
                    if lb is not None:
                        variables_x[i] = lb
                        continue
                    ub = relaxed_problem.upper_bound(var)
                    if ub is not None:
                        variables_x[i] = ub
                        continue
                    variables_x[i] = 0.0

            if not lp_solution.status.is_success():
                break

            for cut in node.storage.cut_node_storage.cuts:
                is_duplicate = False
                for constraint in linear_problem.relaxed.constraints:
                    if constraint.name == cut.name:
                        is_duplicate = True
                if is_duplicate:
                    continue
                expr_tree_data = cut.expr.expression_tree_data(
                    relaxed_problem.num_variables
                )
                fg = expr_tree_data.eval(variables_x)
                fg_x = fg.forward(0, variables_x)[0]
                violated = False
                violation_lb = None
                violation_ub = None

                if cut.lower_bound is not None:
                    if not almost_ge(fg_x, cut.lower_bound, atol=mc.epsilon):
                        violation_lb = fg_x - cut.lower_bound
                        violated = True
                if cut.upper_bound is not None:
                    if not almost_le(fg_x, cut.upper_bound, atol=mc.epsilon):
                        violation_ub = fg_x - cut.upper_bound
                        violated = True

                if violated:
                    logger.debug(
                        run_id,
                        'Cut {} was violated. Adding back to problem. Violation: LB={}; UB={}',
                        cut.name,
                        violation_lb,
                        violation_ub,
                    )
                    num_violated_cuts += 1
                    inherit_cuts_count += 1
                    _add_cut_from_pool_to_problem(linear_problem, cut)

            logger.info(
                run_id,
                'Number of violated cuts at end of loop: {}',
                num_violated_cuts,
            )
        return inherit_cuts_count, lp_solution

    def _try_solve_convex_problem(self, problem):
        """Check if problem is convex in current domain, in that case use
        IPOPT to solve it (if all variables are reals)
        """
        if self._convexity and _is_convex(problem, self._convexity):
            all_reals = all(
                problem.variable_view(v).domain.is_real()
                for v in problem.variables
            )
            if all_reals:
                return self._solve_convex_problem(problem)
        return None

    def _solve_convex_problem(self, problem):
        solution = self._nlp_solver.solve(problem)
        return NodeSolution(solution, solution)

    def _perform_fbbt(self, run_id, problem, tree, node, maxiter=None):
        fbbt_start_time = current_time()
        logger.debug(run_id, 'Performing FBBT')

        objective_upper_bound = None
        if tree.upper_bound is not None:
            objective_upper_bound = tree.upper_bound

        fbbt_maxiter = self.fbbt_maxiter
        if maxiter is not None:
            fbbt_maxiter = maxiter
        branching_variable = None
        if not node.storage.is_root:
            branching_variable = node.storage.branching_variable
        bounds = perform_fbbt(
            problem,
            maxiter=fbbt_maxiter,
            timelimit=self.fbbt_timelimit,
            objective_upper_bound=objective_upper_bound,
            branching_variable=branching_variable,
        )

        self._bounds, self._monotonicity, self._convexity = \
            propagate_special_structure(problem, bounds)

        logger.debug(run_id, 'Set FBBT Bounds')
        cause_infeasibility = None
        for v in problem.variables:
            vv = problem.variable_view(v)
            new_bound = bounds[v]
            if new_bound is None:
                new_bound = Interval(None, None)

            new_lb = best_lower_bound(
                v.domain,
                new_bound.lower_bound,
                vv.lower_bound()
            )

            new_ub = best_upper_bound(
                v.domain,
                new_bound.upper_bound,
                vv.upper_bound()
            )

            if new_lb > new_ub:
                cause_infeasibility = v

        if cause_infeasibility is not None:
            logger.info(
                run_id, 'Bounds on variable {} cause infeasibility',
                cause_infeasibility.name
            )
        else:
            for v in problem.variables:
                vv = problem.variable_view(v)
                new_bound = bounds[v]

                if new_bound is None:
                    new_bound = Interval(None, None)

                new_lb = best_lower_bound(
                    v.domain,
                    new_bound.lower_bound,
                    vv.lower_bound()
                )

                new_ub = best_upper_bound(
                    v.domain,
                    new_bound.upper_bound,
                    vv.upper_bound()
                )

                if np.isinf(new_lb):
                    new_lb = -np.inf

                if np.isinf(new_ub):
                    new_ub = np.inf

                if np.abs(new_ub - new_lb) < mc.epsilon:
                    new_lb = new_ub

                logger.debug(run_id, '  {}: [{}, {}]', v.name, new_lb, new_ub)
                vv.set_lower_bound(new_lb)
                vv.set_upper_bound(new_ub)

        group_name = '_'.join([str(c) for c in node.coordinate])
        logger.tensor(run_id, group_name, 'lb', problem.lower_bounds)
        logger.tensor(run_id, group_name, 'ub', problem.upper_bounds)
        self._bac_telemetry.increment_fbbt_time(
            seconds_elapsed_since(fbbt_start_time)
        )


def _convert_linear_expr(linear_problem, expr, objvar=None):
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
            raise ValueError(
                'Invalid ExpressionType {}'.format(expr.expression_type)
            )

    children = []
    coeffs = []
    for var, coef in coefficients.items():
        children.append(linear_problem.variable(var))
        coeffs.append(coef)

    if objvar is not None:
        children.append(objvar)
        coeffs.append(1.0)

    return core.LinearExpression(children, coeffs, const)


def _is_convex(problem, cvx_map):
    obj = problem.objective
    is_objective_cvx = cvx_map[obj.root_expr].is_convex()

    if not is_objective_cvx:
        return False

    return all(
        _constraint_is_convex(cvx_map, cons)
        for cons in problem.constraints
    )


def _constraint_is_convex(cvx_map, cons):
    cvx = cvx_map[cons.root_expr]
    # g(x) <= UB
    if cons.lower_bound is None:
        return cvx.is_convex()

    # g(x) >= LB
    if cons.upper_bound is None:
        return cvx.is_concave()

    # LB <= g(x) <= UB
    return cvx.is_linear()


def _add_cut_from_pool_to_problem(problem, cut):
    # Cuts from pool are weird because they belong to other problems. So we
    # duplicate the (linear) cuts and insert that instead.
    def _duplicate_expr(expr):
        if expr.is_variable():
            return problem.relaxed.variable(expr.name)
        if isinstance(expr, core.LinearExpression):
            vars = [_duplicate_expr(v) for v in expr.children]
            return core.LinearExpression(vars, expr.linear_coefs, expr.constant_term)
        if isinstance(expr, core.SumExpression):
            children = [_duplicate_expr(ch) for ch in expr.children]
            return core.SumExpression(children)
        if isinstance(expr, core.NegationExpression):
            children = [_duplicate_expr(ch) for ch in expr.children]
            return core.NegationExpression(children)
        if isinstance(expr, core.QuadraticExpression):
            vars1 = [_duplicate_expr(t.var1) for t in expr.terms]
            vars2 = [_duplicate_expr(t.var2) for t in expr.terms]
            coefs = [t.coefficient for t in expr.terms]
            return core.QuadraticExpression(vars1, vars2, coefs)
        raise RuntimeError('Cut contains expr of type {}', type(expr))

    new_expr = _duplicate_expr(cut.expr)
    new_cut = Cut(
        cut.type_,
        cut.name,
        new_expr,
        cut.lower_bound,
        cut.upper_bound,
        cut.is_objective
    )
    return _add_cut_to_problem(problem, new_cut)


def _add_cut_to_problem(problem, cut):
    if not cut.is_objective:
        return problem.add_constraint(
            cut.name,
            cut.expr,
            cut.lower_bound,
            cut.upper_bound,
        )
    else:
        objvar = problem.relaxed.variable('_objvar')
        assert cut.lower_bound is None
        assert cut.upper_bound is None
        new_root_expr = core.SumExpression([
            cut.expr,
            core.LinearExpression([objvar], [-1.0], 0.0)
        ])
        return problem.add_constraint(
            cut.name,
            new_root_expr,
            None,
            0.0
        )
