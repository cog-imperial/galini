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
"""Generic Branch & Bound solver."""
from galini.logging import Logger
from galini.config import (
    SolverOptions,
    NumericOption,
    IntegerOption,
    EnumOption,
)
from galini.solvers import Solver
from galini.cuts import CutsGeneratorsRegistry
from galini.bab.branch_and_cut import BranchAndCutAlgorithm
from galini.util import print_problem


class BranchAndBoundSolver(Solver):
    name = 'bab'

    description = 'Generic Branch & Bound solver.'

    @staticmethod
    def solver_options():
        return SolverOptions(BranchAndBoundSolver.name, [
            NumericOption('tolerance', default=1e-8),
            NumericOption('relative_tolerance', default=1e-8),
            IntegerOption('node_limit', default=100000000),
            IntegerOption('fbbt_maxiter', default=10),
            BranchAndCutAlgorithm.algorithm_options(),
        ])

    def actual_solve(self, problem, run_id, **kwargs):
        algo = BranchAndCutAlgorithm(self.galini)
        return algo.solve(problem, run_id=run_id)