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
from galini.logging import get_logger, DEBUG, WARNING
from galini.util import expr_to_str
from galini.math import mc
from galini.config.options import (
    SolverOptions,
    ExternalSolverOptions,
    OptionsGroup,
    StringOption,
)
from galini.solvers import Solver
from galini.core import (
    ipopt_solve,
    IpoptApplication,
    EJournalLevel,
    PythonJournal,
)
from galini.ipopt.solution import build_solution
from galini.ipopt.logger import IpoptLoggerAdapter


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
        if len(problem.objectives) != 1:
            raise ValueError('Problem must have exactly 1 objective function.')

        if 'ipopt_application' in kwargs:
            app = kwargs.pop('ipopt_application')
        else:
            app = IpoptApplication()
            config = self.galini.get_configuration_group('ipopt')
            self._configure_ipopt_application(app, config)
            self._configure_ipopt_logger(app, config, run_id)

        xi, xl, xu = self.get_starting_point_and_bounds(run_id, problem)
        gl, gu, constraints_root_expr_idxs = \
            self.get_constraints_bounds(run_id, problem)
        logger.debug(run_id, 'Calling in IPOPT')

        tree_data = problem.expression_tree_data()
        out_indexes = [problem.objective.root_expr.idx] + constraints_root_expr_idxs

        ipopt_solution = ipopt_solve(
            app, tree_data, out_indexes, xi, xl, xu, gl, gu, IpoptLoggerAdapter(logger, run_id, DEBUG)
        )
        solution = build_solution(run_id, problem, ipopt_solution, tree_data, out_indexes)
        logger.debug(run_id, 'IPOPT returned {}', solution)
        return solution

    def _configure_ipopt_application(self, app, config):
        config = config['ipopt']
        options = app.options()
        for key, value in config.items():
            if isinstance(value, str):
                options.set_string_value(key, value, True, False)
            elif isinstance(value, int):
                options.set_integer_value(key, value, True, False)
            elif isinstance(value, float):
                options.set_numeric_value(key, value, True, False)

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

        logger.debug(run_id, 'Problem has {} variables', nx)

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
                lb = lb if lb is not None else -2e19

                ub = v.upper_bound()
                ub = ub if ub is not None else 2e19

                xl[i] = lb
                xu[i] = ub

                if v.has_starting_point():
                    xi[i] = v.starting_point()
                else:

                    xi[i] = max(lb, min(ub, 0))
            logger.debug(
                run_id,
                'Variable: {} <= {} <= {}, starting {}',
                xl[i], var.name, xu[i], xi[i])
        return xi, xl, xu

    def get_constraints_bounds(self, run_id, problem):
        ng = problem.num_constraints

        logger.debug(run_id, 'Problem has {} constraints', ng)

        gl = np.zeros(ng)
        gu = np.zeros(ng)
        out_indexes = []
        count = 0
        for i, c in enumerate(problem.constraints):
            if c.metadata.get('rlt_constraint_info'):
                (v, aux_vs) = c.metadata['rlt_constraint_info']
                logger.debug(
                    run_id,
                    'Constraint {}: Skip RLT {} = {}',
                    c.name,
                    v.name,
                    [v.name for v in aux_vs]
                )
                continue
            gl[count] = c.lower_bound if c.lower_bound is not None else -2e19
            gu[count] = c.upper_bound if c.upper_bound is not None else 2e19
            logger.debug(
                run_id,
                'Constraint {}: {} <= {} <= {}',
                c.name,
                gl[i],
                expr_to_str(c.root_expr),
                gu[i],
            )
            out_indexes.append(c.root_expr.idx)
            count += 1

        return gl[:count], gu[:count], out_indexes
