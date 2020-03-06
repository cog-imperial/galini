#  Copyright 2020 Francesco Ceccon
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

"""Generic Branch & Bound algorithm."""

import abc

import pyomo.environ as pe

from galini.algorithms import Algorithm
from galini.branch_and_bound.solution import BabSolution, BabStatusInterrupted
from galini.branch_and_bound.tree import BabTree
from galini.branch_and_cut.node_storage import RootNodeStorage
from galini.config import (
    NumericOption,
    IntegerOption,
    BoolOption,
)
from galini.math import is_close
from galini.quantities import relative_gap, absolute_gap


class BranchAndBoundAlgorithm(Algorithm, metaclass=abc.ABCMeta):
    def __init__(self, galini):
        super().__init__(galini)
        config = galini.get_configuration_group(self.name)
        self._catch_keyboard_interrupt = config.get('catch_keyboard_interrupt', True)
        self._tree = None
        self._solution = None
        self._telemetry = galini.telemetry
        self.logger = galini.get_logger(__name__)

    @staticmethod
    def bab_options():
        return [
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
            NumericOption(
                'branching_weight_sum',
                default=1.0,
                description='Weight of the sum of nonlinear infeasibility'
            ),
            NumericOption(
                'branching_weight_max',
                default=0.0,
                description='Weight of the max of nonlinear infeasibility'
            ),
            NumericOption(
                'branching_weight_min',
                default=0.0,
                description='Weight of the min of nonlinear infeasibility'
            ),
            NumericOption(
                'branching_weight_lambda',
                default=0.25,
                description='Weight of the midpoint when computing convex combination with solution for branching'
            ),

        ]

    @abc.abstractmethod
    def find_initial_solution(self, model, tree, node):
        raise NotImplemented()

    @abc.abstractmethod
    def solve_problem_at_root(self, tree, node):
        raise NotImplemented()

    @abc.abstractmethod
    def solve_problem_at_node(self, tree, node):
        raise NotImplemented()

    @property
    @abc.abstractmethod
    def branching_strategy(self):
        raise NotImplemented()

    @property
    @abc.abstractmethod
    def node_selection_strategy(self):
        raise NotImplemented()

    @property
    @abc.abstractmethod
    def bab_config(self):
        raise NotImplemented()

    def node_limit_exceeded(self, state):
        return state.nodes_visited > self.bab_config['node_limit']

    def has_converged(self, state):
        rel_gap = relative_gap(state.lower_bound, state.upper_bound)
        abs_gap = absolute_gap(state.lower_bound, state.upper_bound)

        bounds_close = is_close(
            state.lower_bound,
            state.upper_bound,
            rtol=self.bab_config['relative_tolerance'],
            atol=self.bab_config['tolerance'],
        )

        if self.galini.paranoid_mode:
            assert (state.lower_bound <= state.upper_bound or bounds_close)

        return (
            rel_gap <= self.bab_config['relative_tolerance'] or
            abs_gap <= self.bab_config['tolerance']
        )

    def should_terminate(self, state):
        return (
            self.node_limit_exceeded(state) or
            self.has_converged(state) or
            self.galini.timelimit.timeout()
        )

    def actual_solve(self, model, **kwargs):
        # Run branch_and_bound loop, catch keyboard interrupt from users
        if self._catch_keyboard_interrupt:
            try:
                self._bab_loop(model, **kwargs)
            except KeyboardInterrupt:
                pass
        else:
            self._bab_loop(model, **kwargs)
        assert self._tree is not None
        return self._solution_from_tree(model, self._tree)

    def _bab_loop(self, model, **kwargs):
        known_optimal_objective = kwargs.get('known_optimal_objective', None)
        if known_optimal_objective is not None:
            if not model.__objective.is_originally_minimizing:
                known_optimal_objective = -known_optimal_objective

        branching_strategy = self.branching_strategy
        node_selection_strategy = self.node_selection_strategy

        bab_iteration = 0

        root_node_storage = RootNodeStorage(model)
        tree = BabTree(
            root_node_storage, branching_strategy, node_selection_strategy
        )
        self._tree = tree

        self.logger.info('Finding initial feasible solution')

        initial_solution = self.find_initial_solution(model, tree, tree.root)

        if initial_solution is not None:
            tree.add_initial_solution(initial_solution, self.galini.mc)
            self._telemetry.log_at_end_of_iteration(bab_iteration)
            if self.should_terminate(tree.state):
                return

        self.logger.info('Solving root problem')
        root_solution = self.solve_problem_at_root(tree, tree.root)
        tree.update_root(root_solution)
        self._telemetry.log_at_end_of_iteration(bab_iteration)
        bab_iteration += 1

        self.logger.info('Root problem solved, tree state {}', tree.state)
        self.logger.info('Root problem solved, root solution {}', root_solution)
        self.logger.log_add_bab_node(
            coordinate=[0],
            lower_bound=root_solution.lower_bound,
            upper_bound=root_solution.upper_bound,
        )

        while not self.should_terminate(tree.state):
            self.logger.info('Tree state at beginning of iteration: {}', tree.state)
            if not tree.has_nodes():
                self.logger.info('No more nodes to visit.')
                break

            current_node = tree.next_node()
            if current_node.parent is None:
                # This is the root node.
                node_children, branching_point = tree.branch_at_node(current_node)
                self.logger.info( 'Branched at point {}', branching_point)
                continue

            self.logger.info(
                'Visiting node {}: parent state={}',
                current_node.coordinate,
                current_node.parent.state,
            )

            node_can_not_improve_solution = is_close(
                current_node.parent.lower_bound,
                tree.upper_bound,
                atol=self.bab_config['tolerance'],
                rtol=self.bab_config['relative_tolerance'],
            ) or current_node.parent.lower_bound > tree.upper_bound

            if node_can_not_improve_solution:
                self.logger.info(
                    "Fathom node because it won't improve bound: node.lower_bound={}, tree.upper_bound={}",
                    current_node.parent.lower_bound,
                    tree.upper_bound,
                )
                self.logger.log_prune_bab_node(current_node.coordinate)
                tree.fathom_node(current_node, update_nodes_visited=True)
                self._telemetry.log_at_end_of_iteration(bab_iteration)
                bab_iteration += 1
                continue

            solution = self.solve_problem_at_node(tree, current_node)
            assert solution is not None

            tree.update_node(current_node, solution)
            self.logger.info('Node {} solution: {}', current_node.coordinate, solution)
            self.logger.log_add_bab_node(
                coordinate=current_node.coordinate,
                lower_bound=solution.lower_bound,
                upper_bound=solution.upper_bound,
            )
            current_node_converged = is_close(
                solution.lower_bound,
                solution.upper_bound,
                atol=self.bab_config['tolerance'],
                rtol=self.bab_config['relative_tolerance'],
            )

            node_relaxation_is_feasible_or_unbounded = (
                    solution.lower_bound_solution is not None and
                    (
                        solution.lower_bound_solution.status.is_success() or
                        solution.lower_bound_solution.status.is_unbounded()
                    )
            )

            print('nxxx')
            print(solution.lower_bound_solution)
            print(solution.upper_bound_solution)
            print(node_relaxation_is_feasible_or_unbounded)
            # input('xxxx')
            if not current_node_converged and node_relaxation_is_feasible_or_unbounded:
                node_children, branching_point = tree.branch_at_node(current_node)
                self.logger.info('Branched at point {}', branching_point)
            else:
                # We won't explore this part of the tree anymore.
                # Add to fathomed nodes.
                self.logger.info(
                    'Fathom node {}, converged? {}, upper_bound_solution {}',
                    current_node.coordinate, current_node_converged, solution.upper_bound_solution
                )
                self.logger.log_prune_bab_node(current_node.coordinate)
                tree.fathom_node(current_node, update_nodes_visited=False)

            self.logger.info('New tree state at {}: {}', current_node.coordinate, tree.state)
            self.logger.update_variable('z_l', tree.nodes_visited, tree.lower_bound)
            self.logger.update_variable('z_u', tree.nodes_visited, tree.upper_bound)
            self.logger.info(
                'Child {} has solutions: LB={} UB={}',
                current_node.coordinate,
                solution.lower_bound_solution,
                solution.upper_bound_solution,
            )
            self._telemetry.log_at_end_of_iteration(bab_iteration)
            bab_iteration += 1

        self.logger.info('Branch & Bound Finished: {}', tree.state)
        self.logger.info('Branch & Bound Converged?: {}', self.has_converged(tree.state))
        self.logger.info('Branch & Bound Timeout?: {}', self.galini.timelimit.timeout())
        self.logger.info('Branch & Bound Node Limit Exceeded?: {}', self.node_limit_exceeded(tree.state))

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
            is_timeout=self.galini.timelimit.timeout(),
            has_converged=self.has_converged(tree.state),
            node_limit_exceeded=self.node_limit_exceeded(tree.state),
        )
