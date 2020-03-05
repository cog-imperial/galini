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
"""Base classes for solvers."""
import datetime
import abc
from galini.timelimit import seconds_left
from galini.cuts import CutsGeneratorsManager
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
    def __init__(self, galini):
        self.galini = galini

    @property
    def config(self):
        return self.galini.get_configuration_group(self.name)

    def before_solve(self, model):
        pass

    def solve(self, model, **kwargs):
        """Solve the optimization problem.

        Arguments
        ---------
        model: ConcreteModel
            the optimization problem
        kwargs: dict
            additional (possibly solver specific) keyword arguments

        Returns
        -------
        Solution
        """
        run_id = _create_run_id(self.name)
        self.galini.logger.log_solve_start(run_id, self.name)
        solution = self.actual_solve(model, run_id=run_id, **kwargs)
        self.galini.logger.log_solve_end(run_id, self.name)
        return solution

    @abc.abstractmethod
    def actual_solve(self, model, run_id, **kwargs):
        pass



class MINLPSolver(Solver):
    """Base class for MINLP solvers."""
    pass
