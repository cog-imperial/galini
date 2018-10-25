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
from typing import Any
from galini.core import Problem
from galini.config import GaliniConfig


class Solver(object):
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
    def __init__(self, config: GaliniConfig,
                 _mip_solver_registry: 'MIPSolverRegistry',
                 _nlp_solver_registry: 'NLPSolverRegistry') -> None:
        pass

    def solve(self, problem: Problem, **kwargs: Any) -> Any:
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
