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
from galini.logging import Logger, INFO, WARNING
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


class IpoptStatus(Status):
    def __init__(self, status):
        self._status = status

    def is_success(self):
        return self._status == CoreIpoptSolution.StatusType.success

    def is_infeasible(self):
        return self._status == CoreIpoptSolution.StatusType.local_infeasibility

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

    def actual_solve(self, problem, **kwargs):
        if len(problem.objectives) != 1:
            raise ValueError('Problem must have exactly 1 objective function.')

        self.logger = Logger.from_kwargs(kwargs)
        if 'ipopt_application' in kwargs:
            app = kwargs.pop('ipopt_application')
            print('Using app = ', app)
        else:
            app = IpoptApplication()
            self._configure_ipopt_application(app, self.config.ipopt)
            self._configure_ipopt_logger(app, self.config.ipopt)

        xi, xl, xu = self.get_starting_point_and_bounds(problem)
        gl, gu = self.get_constraints_bounds(problem)
        self.logger.debug('Calling in IPOPT')

        ipopt_solution = ipopt_solve(
            app, problem, xi, xl, xu, gl, gu, _IpoptLoggerAdapter(self.logger, INFO)
        )
        solution = self._build_solution(problem, ipopt_solution)
        self.logger.debug('IPOPT returned {}', solution)
        return solution

    def _configure_ipopt_application(self, app, config):
        options = app.options()
        for key, value in config.items():
            if isinstance(value, str):
                options.set_string_value(key, value, True, False)
            elif isinstance(value, int):
                options.set_integer_value(key, value, True, False)
            elif isinstance(value, float):
                options.set_numeric_value(key, value, True, False)

    def _configure_ipopt_logger(self, app, config):
        logging_config = config['logging']
        level_name = logging_config.get('level', 'J_ITERSUMMARY')
        level = getattr(EJournalLevel, level_name)
        journal = PythonJournal('Default', level, _IpoptLoggerAdapter(self.logger, INFO))
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

    def get_starting_point_and_bounds(self, problem):
        nx = problem.num_variables

        self.logger.debug('Problem has {} variables', nx)

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
            self.logger.debug(
                'Variable: {} <= {} <= {}, starting {}',
                xl[i], var.name, xu[i], xi[i])
        return xi, xl, xu

    def get_constraints_bounds(self, problem):
        ng = problem.num_constraints

        self.logger.debug('Problem has {} constraints', ng)

        gl = np.zeros(ng)
        gu = np.zeros(ng)
        for i in range(ng):
            c = problem.constraint(i)
            gl[i] = c.lower_bound if c.lower_bound is not None else -2e19
            gu[i] = c.upper_bound if c.upper_bound is not None else 2e19
            self.logger.debug('Constraint: {} <= {} <= {}', gl[i], c.name, gu[i])

        return gl, gu


class _IpoptLoggerAdapter(object):
    def __init__(self, logger, level):
        self._logger = logger
        self._level = level

    def write(self, msg):
        self._logger.log(self._level, msg)

    def flush(self):
        pass
