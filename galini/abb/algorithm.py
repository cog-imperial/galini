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

"""Alpha BB algorithm."""
import numpy as np
from galini.logging import Logger
from galini.bab import (
    BabTree,
    KSectionBranchingStrategy,
    NodeSelectionStrategy,
    NodeSolution,
    BabAlgorithm,
)
from galini.abb.relaxation import AlphaBBRelaxation


class AlphaBBAlgorithm(BabAlgorithm):
    def __init__(self, nlp_solver, minlp_solver, config):
        self.initialize(config)
        self._nlp_solver = nlp_solver
        self._minlp_solver = minlp_solver

    def solve_problem(self, problem):
        self.logger.info('Solving problem {}', problem.name)
        relaxation = AlphaBBRelaxation()
        relaxed_problem = relaxation.relax(problem)
        solution = self._minlp_solver.solve(relaxed_problem, logger=self.logger)

        assert len(solution.objectives) == 1
        relaxed_obj_value = solution.objectives[0].value

        x_value = dict([(v.name, v.value) for v in solution.variables])
        x = [x_value[v.name] for v in problem.variables]

        sol = self._minlp_solver.solve(problem)
        obj_value = sol.objectives[0].value
        # assert np.isclose(relaxed_obj_value, obj_value) and relaxed_obj_value <= obj_value
        return NodeSolution(
            relaxed_obj_value,
            obj_value,
            sol,
        )
