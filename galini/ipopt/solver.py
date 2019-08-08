# Copyright 2019 Francesco Ceccon
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

"""Solve NLP using Ipopt."""

import numpy as np

from galini.config.options import (
    SolverOptions,
    ExternalSolverOptions,
    OptionsGroup,
    StringOption,
)
from galini.core import (
    ipopt_solve,
    IpoptApplication,
    EJournalLevel,
    PythonJournal,
)
from galini.ipopt.logger import IpoptLoggerAdapter
from galini.ipopt.solution import build_solution
from galini.logging import get_logger, DEBUG
from galini.math import mc, is_inf
from galini.pyomo.postprocess import PROBLEM_RLT_CONS_INFO
from galini.solvers import Solver
from galini.timelimit import seconds_left
from galini.util import expr_to_str


logger = get_logger(__name__)


class IpoptNLPSolver(Solver):
    """Solver for NLP problems that uses Ipopt."""
    name = 'ipopt'

    description = 'NLP solver.'

    @staticmethod
    def solver_options():
        return SolverOptions(IpoptNLPSolver.name, [
            ExternalSolverOptions('ipopt'),
            OptionsGroup('logging', [
                StringOption('level', default='J_ITERSUMMARY'),
            ]),
        ])

    def actual_solve(self, problem, run_id, **kwargs):
        config = self.galini.get_configuration_group('ipopt')

        if 'ipopt_application' in kwargs:
            app = kwargs.pop('ipopt_application')
        else:
            app = IpoptApplication()
            self._configure_ipopt_logger(app, config, run_id)

        timelimit = kwargs.pop('timelimit', None)

        if timelimit is None:
            timelimit = seconds_left()
        else:
            # Respect global time limit
            timelimit = min(timelimit, seconds_left())

        self._configure_ipopt_application(app, config, timelimit)

        xi, xl, xu = self.get_starting_point_and_bounds(run_id, problem)
        gl, gu, constraints_root_expr_idxs = \
            self.get_constraints_bounds(run_id, problem)
        objective_root_expr_idxs = [problem.objective.root_expr.idx]

        tree_data = problem.expression_tree_data()
        out_indexes = objective_root_expr_idxs + constraints_root_expr_idxs

        logger_adapter = IpoptLoggerAdapter(logger, run_id, DEBUG)

        if logger.level <= DEBUG:
            self._print_debug_problem(run_id, problem, xi, xl, xu, gl, gu)

        logger.debug(run_id, 'Calling in IPOPT')
        ipopt_solution = ipopt_solve(
            app, tree_data, out_indexes, xi, xl, xu, gl, gu, logger_adapter
        )
        solution = build_solution(
            run_id, problem, ipopt_solution, tree_data, out_indexes
        )
        logger.debug(run_id, 'IPOPT returned {}', solution)
        return solution

    def _configure_ipopt_application(self, app, config, timelimit):
        config = config['ipopt']
        options = app.options()
        for key, value in config.items():
            if isinstance(value, str):
                options.set_string_value(key, value, True, False)
            elif isinstance(value, int):
                options.set_integer_value(key, value, True, False)
            elif isinstance(value, float):
                options.set_numeric_value(key, value, True, False)
        # set time limit
        options.set_numeric_value('max_cpu_time', timelimit, True, False)

    def _configure_ipopt_logger(self, app, config, run_id):
        logging_config = config['logging']
        level_name = logging_config.get('level', 'J_ITERSUMMARY')
        level = getattr(EJournalLevel, level_name)
        journal = PythonJournal('Default', level, IpoptLoggerAdapter(logger, run_id, DEBUG))
        journalist = app.journalist()
        journalist.delete_all_journals()
        journalist.add_journal(journal)

    def get_starting_point_and_bounds(self, run_id, problem):
        nx = problem.num_variables

        xi = np.zeros(nx)
        xl = np.zeros(nx)
        xu = np.zeros(nx)
        for i in range(nx):
            var = problem.variable(i)
            v = problem.variable_view(i)
            if v.is_fixed():
                xl[i] = v.value()
                xu[i] = v.value()
                xi[i] = v.value()
            else:
                lb = v.lower_bound()
                if is_inf(lb):
                    lb = -mc.user_upper_bound
                    logger.warning(
                        run_id,
                        'Variable {} lower bound is infinite: setting to {}',
                        var.name,
                        lb,
                    )

                ub = v.upper_bound()
                if is_inf(ub):
                    ub = mc.user_upper_bound
                    logger.warning(
                        run_id,
                        'Variable {} upper bound is infinite: setting to {}',
                        var.name,
                        ub,
                    )

                xl[i] = lb
                xu[i] = ub

                if v.has_starting_point():
                    starting_point = v.starting_point()
                    if is_inf(starting_point):
                        starting_point = max(lb, min(ub, 0))
                    xi[i] = starting_point
                else:
                    xi[i] = max(lb, min(ub, 0))

        return xi, xl, xu

    def get_constraints_bounds(self, run_id, problem):
        ng = problem.num_constraints

        gl = np.zeros(ng)
        gu = np.zeros(ng)
        out_indexes = []
        count = 0
        for i, c in enumerate(problem.constraints):
            if self._skip_constraint(c):
                (v, aux_vs) = c.metadata[PROBLEM_RLT_CONS_INFO]
                logger.debug(
                    run_id,
                    'Constraint {}: Skip RLT {} = {}',
                    c.name,
                    v.name,
                    [v.name for v in aux_vs]
                )
                continue

            lb = c.lower_bound
            ub = c.upper_bound
            if lb is None:
                lb = -mc.infinity
            if ub is None:
                ub = mc.infinity

            gl[count] = lb
            gu[count] = ub
            out_indexes.append(c.root_expr.idx)
            count += 1

        return gl[:count], gu[:count], out_indexes

    def _skip_constraint(self, constraint):
        return PROBLEM_RLT_CONS_INFO in constraint.metadata

    def _print_debug_problem(self, run_id, problem, xi, xl, xu, gl, gu):
        logger.debug(run_id, 'Num Variables: {}', problem.num_variables)
        logger.debug(run_id, 'Num Constraints: {}', problem.num_constraints)

        logger.debug(run_id, 'Variables:')
        for i, var in enumerate(problem.variables):
            logger.debug(
                run_id,
                '\t{}: [{}, {}] start {}',
                var.name, xl[i], xu[i], xi[i]
            )

        logger.debug(run_id, 'Objective:')
        logger.debug(
            run_id, '\t {}', expr_to_str(problem.objective.root_expr)
        )

        logger.debug(run_id, 'Constraints:')
        idx = 0
        for constraint in problem.constraints:
            if self._skip_constraint(constraint):
                continue
            logger.debug(
                run_id,
                '\t{}: {} <= {} <= {}',
                constraint.name,
                gl[idx], expr_to_str(constraint.root_expr), gu[idx]
            )
            idx += 1
