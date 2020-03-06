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

"""Generic Branch & Bound solver."""
import numpy as np
import pyomo.environ as pe

from galini.branch_and_cut.algorithm import BranchAndCutAlgorithm
from galini.branch_and_bound.solution import BabSolution, BabStatusInterrupted
from galini.branch_and_bound.tree import BabTree
from galini.branch_and_cut.node_storage import RootNodeStorage
from galini.branch_and_cut.telemetry import BranchAndCutTelemetry
from galini.config import (
    SolverOptions,
    NumericOption,
    IntegerOption,
    BoolOption,
)
from galini.logging import get_logger
from galini.math import is_close
from galini.solvers import Solver, OptimalObjective, OptimalVariable
from galini.solvers.solution import load_solution_from_model
from galini.timelimit import elapsed_time


logger = get_logger(__name__)


class BranchAndBoundSolver(Solver):
    name = 'branch_and_cut'

    description = 'Generic Branch & Bound solver.'

    def __init__(self, galini):
        super().__init__(galini)
        config = galini.get_configuration_group(self.name)
        self._catch_keyboard_interrupt = config.get('catch_keyboard_interrupt', True)
        self._tree = None
        self._solution = None
        self._telemetry = galini.telemetry
        self._bac_telemetry = BranchAndCutTelemetry(galini.telemetry)
        self._algo = BranchAndCutAlgorithm(
            galini, solver=self, telemetry=self._bac_telemetry
        )

    @staticmethod
    def solver_options():
        return SolverOptions(BranchAndBoundSolver.name, [
            NumericOption('tolerance', default=1e-6),
            NumericOption('relative_tolerance', default=1e-6),
            IntegerOption('node_limit', default=100000000),
            IntegerOption('root_node_feasible_solution_seed', default=None),
            NumericOption('root_node_feasible_solution_search_timelimit', default=6000000),
            IntegerOption('fbbt_maxiter', default=10),
            IntegerOption('obbt_simplex_maxiter', default=1000),
            NumericOption('obbt_timelimit', default=6000000),
            NumericOption('fbbt_timelimit', default=6000000),
            IntegerOption('fbbt_max_quadratic_size', default=1000),
            IntegerOption('fbbt_max_expr_children', default=1000),
            BoolOption('catch_keyboard_interrupt', default=True),
            BranchAndCutAlgorithm.algorithm_options(),
        ])

    def before_solve(self, model):
        pass

    def actual_solve(self, model, run_id, **kwargs):
        # Run branch_and_bound loop, catch keyboard interrupt from users
        if self._catch_keyboard_interrupt:
            try:
                self._bab_loop(model, run_id, **kwargs)
            except KeyboardInterrupt:
                pass
        else:
            self._bab_loop(model, run_id, **kwargs)
        assert self._tree is not None
        return self._solution_from_tree(model, self._tree)

    def _bab_loop(self, model, run_id, **kwargs):
        known_optimal_objective = kwargs.get('known_optimal_objective', None)
        if known_optimal_objective is not None:
            if not model.__objective.is_originally_minimizing:
                known_optimal_objective = -known_optimal_objective

        self._bac_telemetry.start_timing(
            known_optimal_objective,
            elapsed_time(),
        )

        branching_strategy = self._algo.branching_strategy
        node_selection_strategy = self._algo.node_selection_strategy

        bab_iteration = 0

        root_node_storage = RootNodeStorage(model)
        tree = BabTree(
            root_node_storage, branching_strategy, node_selection_strategy
        )
        self._tree = tree

        logger.info(run_id, 'Finding initial feasible solution')
        initial_solution = self._algo.find_initial_solution(
            run_id, model, tree, tree.root
        )

        if initial_solution is not None:
            tree.add_initial_solution(initial_solution)
            self._bac_telemetry.update_at_end_of_iteration(tree, elapsed_time())
            self._telemetry.log_at_end_of_iteration(run_id, bab_iteration)
            if self._algo.should_terminate(tree.state):
                return

        logger.info(run_id, 'Solving root problem')
        root_solution = self._algo.solve_problem_at_root(run_id, tree, tree.root)
        tree.update_root(root_solution)

        self._bac_telemetry.update_at_end_of_iteration(
            tree, elapsed_time(), update_nodes_visited=False
        )
        self._telemetry.log_at_end_of_iteration(run_id, bab_iteration)
        bab_iteration += 1

        logger.info(run_id, 'Root problem solved, tree state {}', tree.state)
        logger.log_add_bab_node(
            run_id,
            coordinate=[0],
            lower_bound=root_solution.lower_bound,
            upper_bound=root_solution.upper_bound,
        )

        while not self._algo.should_terminate(tree.state):
            logger.info(run_id, 'Tree state at beginning of iteration: {}', tree.state)
            #input('continue...')
            if not tree.has_nodes():
                logger.info(run_id, 'No more nodes to visit.')
                break

            current_node = tree.next_node()
            if current_node.parent is None:
                # This is the root node.
                node_children, branching_point = tree.branch_at_node(current_node)
                logger.info(run_id, 'Branched at point {}', branching_point)
                continue

            logger.info(
                run_id,
                'Visiting node {}: parent state={}',
                current_node.coordinate,
                current_node.parent.state,
            )

            node_can_not_improve_solution = is_close(
                current_node.parent.lower_bound,
                tree.upper_bound,
                atol=self._algo.tolerance,
                rtol=self._algo.relative_tolerance,
            ) or current_node.parent.lower_bound > tree.upper_bound

            if node_can_not_improve_solution:
                logger.info(
                    run_id,
                    "Fathom node because it won't improve bound: node.lower_bound={}, tree.upper_bound={}",
                    current_node.parent.lower_bound,
                    tree.upper_bound,
                )
                logger.log_prune_bab_node(run_id, current_node.coordinate)
                tree.fathom_node(current_node, update_nodes_visited=True)
                self._bac_telemetry.update_at_end_of_iteration(tree, elapsed_time())
                self._telemetry.log_at_end_of_iteration(run_id, bab_iteration)
                bab_iteration += 1
                continue

            solution = self._algo.solve_problem_at_node(run_id, tree, current_node)
            assert solution is not None

            tree.update_node(current_node, solution)
            logger.log_add_bab_node(
                run_id,
                coordinate=current_node.coordinate,
                lower_bound=solution.lower_bound,
                upper_bound=solution.upper_bound,
            )
            current_node_converged = is_close(
                solution.lower_bound,
                solution.upper_bound,
                atol=self._algo.tolerance,
                rtol=self._algo.relative_tolerance,
            )

            if not current_node_converged and solution.upper_bound_solution is not None:
                node_children, branching_point = tree.branch_at_node(current_node)
                logger.info(run_id, 'Branched at point {}', branching_point)
            else:
                # We won't explore this part of the tree anymore.
                # Add to fathomed nodes.
                logger.info(
                    run_id, 'Fathom node {}, converged? {}, upper_bound_solution {}',
                    current_node.coordinate, current_node_converged, solution.upper_bound_solution
                )
                logger.log_prune_bab_node(run_id, current_node.coordinate)
                tree.fathom_node(current_node, update_nodes_visited=False)

            # self._log_problem_information_at_node(run_id, current_node.storage.problem, solution, current_node)

            logger.info(run_id, 'New tree state at {}: {}', current_node.coordinate, tree.state)
            logger.update_variable(run_id, 'z_l', tree.nodes_visited, tree.lower_bound)
            logger.update_variable(run_id, 'z_u', tree.nodes_visited, tree.upper_bound)
            logger.info(
                run_id,
                'Child {} has solutions: LB={} UB={}',
                current_node.coordinate,
                solution.lower_bound_solution,
                solution.upper_bound_solution,
            )
            self._bac_telemetry.update_at_end_of_iteration(tree, elapsed_time())
            self._telemetry.log_at_end_of_iteration(run_id, bab_iteration)
            bab_iteration += 1

        logger.info(
            run_id, 'Branch & Bound Finished: {}', tree.state)
        logger.info(
            run_id,
            'Branch & Bound Converged?: {}',
            self._algo._has_converged(tree.state))
        logger.info(
            run_id, 'Branch & Bound Timeout?: {}', self._algo._timeout())
        logger.info(
            run_id,
            'Branch & Bound Node Limit Exceeded?: {}',
            self._algo._node_limit_exceeded(tree.state))

    def _solution_from_tree(self, problem, tree):
        nodes_visited = tree.nodes_visited

        if len(tree.solution_pool) == 0:
            # Return lower bound only
            optimal_vars = pe.ComponentMap(
                (v, pe.value(v))
                for v in problem.component_data_objects(pe.Var, active=True)
            )
            return BabSolution(
                BabStatusInterrupted(),
                None,
                optimal_vars,
                dual_bound=tree.state.lower_bound,
                nodes_visited=nodes_visited,
            )

        primal_solution = tree.solution_pool.head

        return BabSolution(
            primal_solution.status,
            primal_solution.objective,
            primal_solution.variables,
            dual_bound=tree.state.lower_bound,
            nodes_visited=nodes_visited,
            nodes_remaining=len(tree.open_nodes),
            is_timeout=self._algo._timeout(),
            has_converged=self._algo._has_converged(tree.state),
            node_limit_exceeded=self._algo._node_limit_exceeded(tree.state),
        )

    def _log_problem_information_at_node(self, run_id, problem, solution, node):
        group_name = '_'.join([str(c) for c in node.coordinate])
        logger.tensor(
            run_id,
            group=group_name,
            dataset='lower_bounds',
            data=np.array(problem.lower_bounds)
        )
        logger.tensor(
            run_id,
            group=group_name,
            dataset='upper_bounds',
            data=np.array(problem.upper_bounds)
        )
        solution = solution.upper_bound_solution
        if solution is None:
            return
        if solution.status.is_success():
            logger.tensor(
                run_id,
                group=group_name,
                dataset='solution',
                data=np.array([v.value for v in solution.variables]),
            )
