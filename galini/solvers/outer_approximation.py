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
"""Outer-Approximation solver for Convex MINLP problems."""

import numpy as np
import pulp
from galini.core import Domain
from galini.solvers import MINLPSolver
from galini.relaxations import ContinuousRelaxation
from galini.solvers.solution import OptimalObjective, OptimalVariable, Status, Solution
import  galini.logging as log
from galini.__version__ import __version__
import logging


class PulpStatus(Status):
    def __init__(self, inner):
        self._inner = inner

    def is_success(self):
        return self._inner == pulp.LpStatusOptimal

    def description(self):
        return pupl.LpStatus[self._inner]


class OuterApproximationAlgorithm(object):
    def __init__(self, nlp_solver, solver_name, run_id):
        self._nlp_solver = nlp_solver
        self._tolerance = 1e-5

        self._iter = 1
        self._maximum_iterations = 10
        self._solver_name = solver_name
        self._run_id = run_id

    def _complete_iteration(self):
        self._iter += 1

    def _maximum_iterations_reached(self):
        return self._iter > self._maximum_iterations

    def _log_info(self, msg, *args, **kwargs):
        log.info(self._solver_name, self._run_id, msg, *args, **kwargs)

    def _log_tensor(self, group, dataset, data):
        group_name = 'outer_approximation/{}'.format(self._iter)
        if group is not None:
            group_name += '/' + str(group)
        log.tensor(self._solver_name, self._run_id, group_name, dataset, data)

    def solve_continuous_relaxation(self, problem):
        relaxation = ContinuousRelaxation()
        relaxed_problem = relaxation.relax(problem)
        return self._nlp_solver.solve(relaxed_problem)

    def build_linear_relaxation(self, problem, x_k):
        log = logging.getLogger(pulp.__name__)
        log.setLevel(4500) # silence pulp
        def _build_variable(var):
            assert var.lower_bound is not None
            assert var.upper_bound is not None
            domain = pulp.LpContinuous
            if var.domain == Domain.INTEGER:
                domain = pulp.LpInteger
            elif var.domain == Domain.BINARY:
                domain = pulp.LpBinary
            return pulp.LpVariable(var.name, var.lower_bound, var.upper_bound, domain)

        num_var = problem.num_variables
        num_obj = problem.num_objectives
        num_con = problem.num_constraints
        f_idx = [f.root_expr.idx for f in problem.objectives]
        g_idx = [g.root_expr.idx for g in problem.constraints]
        fg = problem.expression_tree_data().eval(x_k, f_idx + g_idx)
        fg_x = fg.forward(0, x_k)
        w = np.zeros(num_obj + num_con)

        lp = pulp.LpProblem(problem.name + '_lp', pulp.LpMinimize)

        x = [_build_variable(v) for v in problem.variables]
        alpha = pulp.LpVariable('alpha')

        # objective: minimize alpha
        lp += alpha

        # build constraints
        for i in range(num_obj + num_con):
            # compute derivative of f_i(x)
            w[i] = 1.0
            d_fg = fg.reverse(1, w)
            w[i] = 0.0

            expr = np.dot(d_fg, x - x_k) + fg_x[i]

            if i <= num_obj:
                lp += expr <= alpha
            else:
                cons = problem.constraints[i-num_obj]
                if cons.lower_bound is not None:
                    # f(x) >= lb -> -f(x) <= lb
                    lp += expr >= cons.lower_bound
                if cons.upper_bound is not None:
                    lp += expr <= cons.upper_bound
        return lp, alpha, x

    def solve_linear_relaxation(self, problem, x_k):
        lp, alpha, xs = self.build_linear_relaxation(problem, x_k)
        status = lp.solve()

        return Solution(
            PulpStatus(status),
            [OptimalObjective(obj.name, pulp.value(alpha)) for obj in problem.objectives],
            [OptimalVariable(v.name, pulp.value(xs[i])) for i, v in enumerate(problem.variables)],
        )

    def log_header(self):
        self._log_info("""
This is GALINI outer_approximation.

Version: {}
        """.format(__version__))

    def log_problem_summary(self, problem):
        self._log_info("""
Number of variables: {}
Number of constraints: {}
""".format(problem.num_variables, problem.num_constraints))

    def log_iter_summary_header(self):
        self._log_info('\t iter \t z_l \t z_u \t epsilon\n')

    def log_iter_summary(self, z_l, z_u):
        self._log_info('\t {} \t {} \t {} \t {}\n'.format(self._iter, z_l, z_u, z_u - z_l))

    def solve(self, problem):
        self.log_header()
        self.log_problem_summary(problem)

        self._log_info("Solving continuous relaxation.")
        solution = self.solve_continuous_relaxation(problem)
        obj_upper = np.inf
        obj_lower = -np.inf
        linear_relax_feasible = True
        x_k = np.zeros(problem.num_variables, dtype=np.float)

        self.log_iter_summary_header()

        while (obj_upper - obj_lower) > self._tolerance and linear_relax_feasible:
            if self._maximum_iterations_reached():
                break

            for i, sol in enumerate(solution.variables):
                x_k[i] = sol.value

            solution = self.solve_linear_relaxation(problem, x_k)

            assert len(solution.objectives)
            obj_lower = solution.objectives[0].value

            child = problem.make_child()
            for i, v in enumerate(problem.variables):
                if v.domain != Domain.REAL:
                    child.fix(v, solution.variables[i].value)

            solution = self._nlp_solver.solve(child)
            if not solution.status.is_success():
                raise NotImplementedError()
            obj_upper = min(obj_upper, solution.objectives[0].value)

            self.log_iter_summary(obj_lower, obj_upper)
            self._complete_iteration()

        self._log_info('Number of iterations: {}'.format(self._iter))

        return solution


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
        return algo.solve(problem)
