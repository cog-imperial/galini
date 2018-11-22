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
from galini.core import Problem


class Solver(object):
    name = None
    """Base class for all solvers.

    Arguments
    ---------
    config: GaliniConfig
       Galini configuration file.
    mip_solver_registry: MIPSolverRegistry
       registry of available MIP solvers.
    nl_solver_registry: NLPSolverRegistry
       registry of available NLP solvers.
    """
    def __init__(self, config, _mip_solver_registry, _nlp_solver_registry):
        pass

    @property
    def run_id(self):
        """Solver run id."""
        if not getattr(self, '_run_id', None):
            dt = datetime.datetime.utcnow()
            self._run_id = self.name + '_' + dt.strftime('%Y%m%d_%H%M%S')
        return self._run_id

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
        pass


class MINLPSolver(Solver):
    """Base class for MINLP solvers."""
    pass
