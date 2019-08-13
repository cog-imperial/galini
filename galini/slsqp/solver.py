#  Copyright 2019 Francesco Ceccon
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import numpy as np
import scipy.optimize as optimize

from galini.config.options import (
    SolverOptions,
    ExternalSolverOptions,
    OptionsGroup,
    StringOption,
)
from galini.core import (
    ipopt_solve,
    IpoptApplication,
    EJournalLevel,
    PythonJournal,
)
from galini.ipopt.logger import IpoptLoggerAdapter
from galini.ipopt.solution import build_solution
from galini.logging import get_logger, DEBUG
from galini.math import mc, is_inf, is_close
from galini.pyomo.postprocess import PROBLEM_RLT_CONS_INFO
from galini.solvers import Solver
from galini.timelimit import seconds_left, timeout
from galini.solvers.solution import OptimalVariable, OptimalObjective
from galini.slsqp.solution import (
    SlsqpSolution,
    SlsqpSolutionFailure,
    SlsqpSolutionSuccess,
)


logger = get_logger(__name__)


class _ProblemEval:
    def __init__(self, problem, xs, retape=None):
        self.tree_data = problem.expression_tree_data()
        self.constraint_idxs = [
            c.root_expr.idx
            for c in problem.constraints
        ]
        self.objective_idxs = [problem.objective.root_expr.idx]

        self.retape = retape
        self._num_variables = problem.num_variables
        self._num_constraints = problem.num_constraints
        self._objective_eval = None
        self._constraints_eval = None
        self._retape(xs)

    def _retape(self, xs):
        self._objective_eval = self.tree_data.eval(xs, self.objective_idxs)
        self._constraints_eval = self.tree_data.eval(xs, self.constraint_idxs)

    def eval_objective(self, xs):
        if self.retape:
            self._retape(xs)
        f_xs = self._objective_eval.forward(0, xs)
        return f_xs[0]

    def eval_objective_jacobian(self, xs):
        f_xs_jac = self._objective_eval.jacobian(xs)
        return f_xs_jac

    def eval_constraint(self, xs, i, bound, is_ineq, change_sign):
        g_xs = self._constraints_eval.forward(0, xs)
        if not is_ineq:
            return g_xs[i] - bound
        if change_sign:
            return -g_xs[i] - bound
        return g_xs[i] - bound

    def eval_constraint_jacobian(self, xs, i, bound, is_ineq, change_sign):
        n = xs.shape[0]
        g_xs_jac = self._constraints_eval.jacobian(xs)
        # g_xs_jac is a linearized matrix of size num_var * num_cons
        start_idx = i * self._num_variables
        end_idx = (i + 1) * self._num_variables
        return g_xs_jac[start_idx:end_idx]


class SlsqpSolver(Solver):
    """Use Sequential Least Squares Programming to solve non linear problems.

    This class uses SLSQP as implemented by Scipy, see:
    https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html#optimize-minimize-slsqp
    """
    name = "slsqp"

    description = "NLP solver using Sequential Least Squares method."

    def __init__(self, galini):
        super().__init__(galini)

    @staticmethod
    def solver_options():
        """Slsqp solver options."""
        return SolverOptions(SlsqpSolver.name, [])

    def actual_solve(self, problem, run_id, **kwargs):
        logger.debug(run_id, 'Solve NLP with SLSQP')

        starting_point, bounds = self._get_starting_point_and_bounds(problem)
        problem_eval = _ProblemEval(problem, starting_point)
        constraints = self._get_constraints(problem, problem_eval)

        result = None
        with timeout(seconds_left(), 'Timeout in SLSQP'):
            result = optimize.minimize(
                method='SLSQP',
                fun=problem_eval.eval_objective,
                x0=starting_point,
                bounds=bounds,
                jac=problem_eval.eval_objective_jacobian,
                constraints=constraints,
            )

        if result is None:
            status = SlsqpSolutionFailure('Timeout in SLSQP')
            objective = OptimalObjective(problem.objective.name, -np.inf)
            variables = []
            return SlsqpSolution(
                status, objective, variables,
            )

        if result.success:
            status = SlsqpSolutionSuccess(result.message)
        else:
            status = SlsqpSolutionFailure(result.message)

        objective = OptimalObjective(problem.objective.name, result.fun)
        variables = [
            OptimalVariable(v.name, result.x[i])
            for i, v in enumerate(problem.variables)
        ]

        return SlsqpSolution(
            status, objective, variables,
        )


    def _get_starting_point_and_bounds(self, problem):
        x0 = np.zeros(problem.num_variables)
        bounds = [(None, None)] * problem.num_variables

        for i, var in enumerate(problem.variables):
            vv = problem.variable_view(var)

            if vv.is_fixed():
                value = vv.value()
                x0[i] = value
                bounds[i] = (value, value)
            else:
                numerical_lb = lb = vv.lower_bound()

                if is_inf(lb):
                    lb = None
                    numerical_lb = -mc.user_upper_bound

                numerical_ub = ub = vv.upper_bound()
                if is_inf(ub):
                    ub = None
                    numerical_ub = mc.user_upper_bound

                bounds[i] = (lb, ub)

                if vv.has_starting_point():
                    starting_point = vv.starting_point()
                    if is_inf(starting_point):
                        starting_point = max(
                            numerical_lb,
                            min(numerical_ub, 0)
                        )
                    x0[i] = starting_point
                else:
                    x0[i] = max(
                        numerical_lb,
                        min(numerical_ub, 0)
                    )

        return x0, bounds

    def _get_constraints(self, problem, problem_eval):
        constraints = []
        for i, constraint in enumerate(problem.constraints):
            lb = constraint.lower_bound
            ub = constraint.upper_bound

            if lb is None or is_inf(lb):
                constraints.append({
                    "type": "ineq",
                    "fun": problem_eval.eval_constraint,
                    "jac": problem_eval.eval_constraint_jacobian,
                    "args": [i, ub, True, True],
                })
            elif ub is None or is_inf(ub):
                constraints.append({
                    "type": "ineq",
                    "fun": problem_eval.eval_constraint,
                    "jac": problem_eval.eval_constraint_jacobian,
                    "args": [i, lb, True, False],
                })
            elif is_close(lb, ub, atol=mc.epsilon):
                constraints.append({
                    "type": "eq",
                    "fun": problem_eval.eval_constraint,
                    "jac": problem_eval.eval_constraint_jacobian,
                    "args": [i, lb, False, False],
                })
            else:
                # Constraint has both lower and upper bounds, but it's not
                # an equality constraint.
                constraints.append({
                    "type": "ineq",
                    "fun": problem_eval.eval_constraint,
                    "jac": problem_eval.eval_constraint_jacobian,
                    "args": [i, lb, True, False],
                })
                constraints.append({
                    "type": "ineq",
                    "fun": problem_eval.eval_constraint,
                    "jac": problem_eval.eval_constraint_jacobian,
                    "args": [i, ub, True, True],
                })

        return constraints