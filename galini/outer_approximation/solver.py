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
"""Outer Approximation solver."""

import numpy as np
from galini.solvers import MINLPSolver
from galini.special_structure import detect_special_structure
from galini.relaxations import ContinuousRelaxation
from galini.outer_approximation.algorithm import OuterApproximationAlgorithm


# TODO(fra): add options to check convexity of problem before solve
class OuterApproximationSolver(MINLPSolver):
    """Solver implementing Outer-Appoximation for Convex MINLP problems.

    References
    ----------
    [0] Bonami P., Biegler L. T., Conn A. R., Cornuéjols G., Grossmann I. E., Laird C. D.,
        Lee J, Lodi A., Margot F., Sawaya N., Wächter A. (2008).
        An algorithmic framework for convex mixed integer nonlinear programs.
        Discrete Optimization
        https://doi.org/10.1016/J.DISOPT.2006.10.011
    [1] Duran, M. A., Grossmann, I. E. (1986).
        An outer-approximation algorithm for a class of mixed-integer nonlinear programs.
        Mathematical Programming
        https://doi.org/10.1007/BF02592064
    """
    name = 'outer_approximation'

    description = 'Outer-Approximation for convex MINLP.'

    def __init__(self, config, mip_solver_registry, nlp_solver_registry):
        super().__init__(config, mip_solver_registry, nlp_solver_registry)
        self._nlp_solver_cls = nlp_solver_registry.get('ipopt')
        if self._nlp_solver_cls is None:
            raise RuntimeError('ipopt solver is required for OuterApproximationSolver')
        self._config = config

    def solve(self, problem, **kwargs):
        nlp_solver = self._nlp_solver_cls(self._config, None, None)
        algo = OuterApproximationAlgorithm(nlp_solver, self.name, self.run_id)

        ctx = detect_special_structure(problem)
        for v in problem.variables:
            vv = problem.variable_view(v)
            new_bound = ctx.bounds[v]
            vv.set_lower_bound(new_bound.lower_bound)
            vv.set_upper_bound(new_bound.upper_bound)

        starting_point = self._starting_point(nlp_solver, problem)
        return algo.solve(problem, starting_point=starting_point)

    def _starting_point(self, nlp_solver, problem):
        continuous_relax = ContinuousRelaxation()
        relaxed = continuous_relax.relax(problem)
        solution = nlp_solver.solve(relaxed)
        return np.array([v.value for v in solution.variables])
