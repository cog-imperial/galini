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
"""Outer-Approximation algorithm for Convex MINLP problems."""

import numpy as np
import pulp
from galini.core import (
    Domain,
    Variable,
    Domain,
    Sense,
    Constraint,
    Objective,
    LinearExpression,
    SumExpression,
)
from galini.logging import Logger
from galini.relaxations import Relaxation, RelaxationResult, ContinuousRelaxation
from galini.__version__ import __version__
from galini.solvers import Solution, Status, OptimalObjective, OptimalVariable
from galini.outer_approximation.milp_relaxation import MilpRelaxation
from galini.outer_approximation.feasibility_problem import FeasibilityProblemRelaxation
from galini.outer_approximation.continuous_relaxation import FixedIntegerContinuousRelaxation


class State(object):
    def __init__(self, z_u=None, z_l=None):
        if z_u is None:
            z_u = np.inf
        self.z_u = z_u

        if z_l is None:
            z_l = -np.inf
        self.z_l = z_l

        self.iteration = 0

    def __str__(self):
        return 'State(iteration={}, z_l={}, z_u={})'.format(
            self.iteration, self.z_l, self.z_u
        )

class OuterApproximationAlgorithm(object):
    def __init__(self, nlp_solver, mip_solver, config):
        self._nlp_solver = nlp_solver
        self._mip_solver = mip_solver
        self.not_success_is_infeasible = True

        self.tolerance = config['tolerance']
        self._maximum_iterations = config['maxiter']

    def solve(self, problem, **kwargs):
        starting_point = kwargs.pop('starting_point', None)
        if starting_point is None:
            raise ValueError('"starting_point" must be specified')

        self.logger = Logger.from_kwargs(kwargs)

        state = State()
        solutions = [starting_point]

        fixed_integer_relaxation = FixedIntegerContinuousRelaxation()
        milp_relaxation = MilpRelaxation()
        feasibility_problem_relaxation = FeasibilityProblemRelaxation()

        p_oa_t = None

        p_x = fixed_integer_relaxation.relax(problem, x_k=starting_point)

        while self._converged(state) and self._iterations_exceeded(state):
            self.logger.info('Starting Iteration: {}', state)
            # Solve P_OA(T) to obtain (alpha, x)
            if p_oa_t is None:
                p_oa_t = milp_relaxation.relax(problem, x_k=starting_point)
            else:
                milp_relaxation.update_relaxation(problem, p_oa_t, x_k=x_k)
            p_oa_t_solution = self._mip_solver.solve(p_oa_t.lp, logger=self.logger)
            assert np.isfinite(p_oa_t.alpha.value())

            # update lower bound
            state.z_l = p_oa_t.alpha.value()
            self.logger.update_variable('z_l', state.iteration, state.z_l)

            # Solve P_x
            fixed_integer_relaxation.update_relaxation(
                problem,
                p_x,
                x_k=np.array([v.value() for v in p_oa_t.x])
            )
            p_x_solution = self._nlp_solver.solve(p_x, logger=self.logger)

            if p_x_solution.status.is_success():
                # update upper bound
                assert len(p_x_solution.objectives) == 1
                x_k = np.array([v.value for v in p_x_solution.variables])
                obj_value = p_x_solution.objectives[0].value
                assert np.isfinite(obj_value)
                state.z_u = min(state.z_u, obj_value)
                self.logger.update_variable('z_u', state.iteration, state.z_u)
                self.logger.info('P_X solution was successful, new z_u = {}', state.z_u)
            elif p_x_solution.status.is_infeasible() or self.not_success_is_infeasible:
                self.logger.info(
                    'P_X was not successful: {}. Solving feasibilty',
                    p_x_solution.status.description()
                )
                # solve feasibility problem
                feasibility_problem = feasibility_problem_relaxation.relax(p_x)
                feasibility_solution = self._nlp_solver.solve(
                    feasibility_problem,
                    logger=self.logger,
                )
                # feasibility problem cannot be infeasible
                assert feasibility_solution.status.is_success()
                variables_value = dict([(v.name, v.value) for v in feasibility_solution.variables])
                x_k = np.array([variables_value[v.name] for v in p_x.variables])
            else:
                raise RuntimeError(
                    'P_x solution is not success or infeasible: {}'.format(
                        p_x_solution.status.description())
                )

            self.logger.info('Iteration Completed: {}', state)
            state.iteration += 1
        return self._build_solution_from_p_oa_t(problem, p_oa_t, p_oa_t_solution)

    def _converged(self, state):
        return (state.z_u - state.z_l) > self.tolerance

    def _iterations_exceeded(self, state):
        return state.iteration < self._maximum_iterations

    def _build_solution_from_p_oa_t(self, problem, p_oa_t, solution):
        status = solution.status
        obj_name = problem.objectives[0].name
        opt_obj = [
            OptimalObjective(name=obj_name, value=p_oa_t.alpha.value())
        ]
        opt_vars = [
            OptimalVariable(name=variable.name, value=p_oa_t.x[i].value())
            for i, variable in enumerate(problem.variables)
        ]
        return Solution(status, optimal_obj=opt_obj, optimal_vars=opt_vars)
