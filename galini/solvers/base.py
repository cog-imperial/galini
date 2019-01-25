# Copyright 2017 Francesco Ceccon
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
"""Base classes for solvers."""
import datetime
import abc
from galini.logging import Logger
from galini.core import Problem


def _create_run_id(solver):
    now = datetime.datetime.utcnow()
    return '{}_{}'.format(solver, now.strftime('%Y%m%d%H%M%S'))


class Solver(metaclass=abc.ABCMeta):
    name = None
    """Base class for all solvers.

    Arguments
    ---------
    config : GaliniConfig
        Galini configuration
    solver_registry : SolverRegistry
        Registry of available solvers
    """
    def __init__(self, config, solver_registry):
        self.config = config
        self.solver_registry = solver_registry

    def get_solver(self, solver_name):
        """Get solver from the registry."""
        solver_cls = self.solver_registry.get(solver_name, None)
        if solver_cls is None:
            raise ValueError('No solver "{}"'.format(solver_name))
        return solver_cls

    def instantiate_solver(self, solver_name):
        """Get and instantiate solver from the registry."""
        solver_cls = self.get_solver(solver_name)
        return solver_cls(self.config, self.solver_registry)

    def solve(self, problem, **kwargs):
        """Solve the optimization problem.

        Arguments
        ---------
        problem: ProblemDag
            the optimization problem
        kwargs: dict
            additional (possibly solver specific) keyword arguments

        Returns
        -------
        Solution
        """
        logger = Logger.from_kwargs(kwargs)
        run_id = _create_run_id(self.name)
        with logger.child_logger(solver=self.name, run_id=run_id) as child_logger:
            child_logger.log_solve_start()
            solution = self.actual_solve(problem, logger=child_logger, **kwargs)
            child_logger.log_solve_end()
            return solution

    @abc.abstractmethod
    def actual_solve(self, problem, **kwargs):
        pass


class MINLPSolver(Solver):
    """Base class for MINLP solvers."""
    pass
