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
"""Solve NLP using Ipopt."""
import numpy as np
from galini.logging import get_logger, DEBUG, WARNING
from galini.util import expr_to_str
from galini.config.options import (
    SolverOptions,
    ExternalSolverOptions,
    OptionsGroup,
    StringOption,
)
from galini.solvers import (
    Solver,
    Solution,
    Status,
    OptimalObjective,
    OptimalVariable,
)
from galini.core import (
    ipopt_solve,
    IpoptSolution as CoreIpoptSolution,
    IpoptApplication,
    EJournalLevel,
    PythonJournal,
)


logger = get_logger(__name__)


class IpoptStatus(Status):
    def __init__(self, status):
        self._status = status

    def is_success(self):
        return (
            self._status == CoreIpoptSolution.StatusType.success or
            self._status == CoreIpoptSolution.StatusType.stop_at_acceptable_point
        )

    def is_infeasible(self):
        return self._status == CoreIpoptSolution.StatusType.local_infeasibility

    def is_iterations_exceeded(self):
        return self._status == CoreIpoptSolution.StatusType.maxiter_exceeded

    def is_unbounded(self):
        return False

    def description(self):
        return str(self._status)


class IpoptSolution(Solution):
    def __init__(self, status, optimal_obj=None, optimal_vars=None,
                 zl=None, zu=None, g=None, lambda_=None):
        super().__init__(status, optimal_obj, optimal_vars)
        self.zl = zl
        self.zu = zu
        self.g = g
        self.lambda_ = lambda_

    def __str__(self):
        return 'IpoptSolution(status={}, objective={}, variables={})'.format(
            self.status.description(),
            self.objectives,
            self.variables,
        )

    def __repr__(self):
        return '<{} at {}>'.format(str(self), hex(id(self)))


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
            print('Using app = ', app)
        else:
            app = IpoptApplication()
            config = self.galini.get_configuration_group('ipopt')
            self._configure_ipopt_application(app, config)
            self._configure_ipopt_logger(app, config, run_id)

        xi, xl, xu = self.get_starting_point_and_bounds(run_id, problem)
        gl, gu = self.get_constraints_bounds(run_id, problem)
        logger.debug(run_id, 'Calling in IPOPT')

        ipopt_solution = ipopt_solve(
            app, problem, xi, xl, xu, gl, gu, _IpoptLoggerAdapter(logger, run_id, DEBUG)
        )
        solution = self._build_solution(problem, ipopt_solution)
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
        journal = PythonJournal('Default', level, _IpoptLoggerAdapter(logger, run_id, DEBUG))
        journalist = app.journalist()
        journalist.delete_all_journals()
        journalist.add_journal(journal)

    def _build_solution(self, problem, solution):
        status = IpoptStatus(solution.status)
        opt_obj = OptimalObjective(
            name=problem.objectives[0].name,
            value=solution.objective_value)
        opt_vars = [
            OptimalVariable(name=variable.name, value=solution.x[i])
            for i, variable in enumerate(problem.variables)
        ]
        return IpoptSolution(
            status,
            optimal_obj=opt_obj,
            optimal_vars=opt_vars,
            zl=np.array(solution.zl),
            zu=np.array(solution.zu),
            g=np.array(solution.g),
            lambda_=np.array(solution.lambda_)
        )

    def _build_optimal_variable(self, i, variable, solution):
        OptimalVariable(name=variable.name, value=solution.x[i])

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
        for i in range(ng):
            c = problem.constraint(i)
            gl[i] = c.lower_bound if c.lower_bound is not None else -2e19
            gu[i] = c.upper_bound if c.upper_bound is not None else 2e19
            logger.debug(
                run_id,
                'Constraint {}: {} <= {} <= {}',
                c.name,
                gl[i],
                expr_to_str(c.root_expr),
                gu[i],
            )

        return gl, gu


class _IpoptLoggerAdapter(object):
    def __init__(self, logger, run_id, level):
        self._logger = logger
        self._run_id = run_id
        self._level = level

    def write(self, msg):
        self._logger.log(self._run_id, self._level, msg)

    def flush(self):
        pass
