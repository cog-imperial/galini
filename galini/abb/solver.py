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
"""AlphaBB Solver."""
from galini.solvers import MINLPSolver, SolversRegistry
from galini.abb.algorithm import AlphaBBAlgorithm


class AlphaBBSolver(MINLPSolver):
    name = 'alpha_bb'

    describe = 'AlphaBB for nonconvex MINLP.'

    def __init__(self, config, mip_solver_registry, nlp_solver_registry):
        super().__init__(config, mip_solver_registry, nlp_solver_registry)

        self._nlp_solver_registry = nlp_solver_registry
        self._nlp_solver_cls = nlp_solver_registry.get('ipopt')

        registry = SolversRegistry()
        self._cvx_minlp_solver_cls = registry.get('oa')

        if self._nlp_solver_cls is None:
            raise RuntimeError('ipopt solver is required for AlphaBBSolver')

        if self._cvx_minlp_solver_cls is None:
            raise RuntimeError('outer_approximation solver is required for AlphaBBSolver')

        self._config = config

    def solve(self, problem, **kwargs):
        nlp_solver = self._nlp_solver_cls(self._config, None, self._nlp_solver_registry)
        minlp_solver = self._cvx_minlp_solver_cls(self._config, None, self._nlp_solver_registry)
        algo = AlphaBBAlgorithm(nlp_solver, minlp_solver, self.name, self.run_id)
        return algo.solve(problem)
