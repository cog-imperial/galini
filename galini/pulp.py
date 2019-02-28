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
"""Interface GALINI with Coin-OR pulp."""
import pulp
import logging
import numpy as np
from suspect.expression import ExpressionType
from galini.core import Problem, Sense, Domain
from galini.logging import Logger, INFO, WARNING, ERROR
from galini.timelimit import seconds_left
from galini.solvers import (
    Solver,
    OptimalObjective,
    OptimalVariable,
    Status,
    Solution,
)
from galini.config import (
    SolverOptions,
    ExternalSolverOptions,
)

log = logging.getLogger(pulp.__name__)
log.setLevel(9001) # silence pulp


class PulpStatus(Status):
    def __init__(self, inner):
        self._inner = inner

    def is_success(self):
        return self._inner == pulp.LpStatusOptimal

    def is_infeasible(self):
        return self._inner == pulp.LpStatusInfeasible

    def is_unbounded(self):
        return self._inner == pulp.LpStatusUnbounded

    def description(self):
        return pulp.LpStatus[self._inner]

    def __str__(self):
        return 'PulpStatus({})'.format(self.description())

    def __repr__(self):
        return '<{} at {}>'.format(str(self), hex(id(self)))


class MIPSolution(Solution):
    def __init__(self, status, optimal_obj=None, optimal_vars=None,
                 dual_values=None):
        super().__init__(status, optimal_obj, optimal_vars)
        self.dual_values = dual_values


class MIPSolver(Solver):
    name = 'mip'

    description = 'MIP Solver that delegates to Cplex or CBC.'

    @staticmethod
    def solver_options():
        return SolverOptions(MIPSolver.name, [
            ExternalSolverOptions('cplex'),
        ])

    def actual_solve(self, problem, **kwargs):
        logger = Logger.from_kwargs(kwargs)
        solver = _solver(logger, self.config.mip)
        if isinstance(problem, Problem):
            assert len(problem.objectives) == 1

            pulp_problem, variables = _dag_to_pulp(problem)
            status =  pulp_problem.solve(solver)
            pulp_cons_name = [cons for cons in pulp_problem.constraints]

            optimal_variables = [
                OptimalVariable(var.name, pulp.value(variables[var.idx]))
                for var in problem.variables
            ]

            dual_values = [cons.pi for cons in pulp_problem.constraints.values()]
            if all(dv is None for dv in dual_values):
                dual_values = None

            return MIPSolution(
                PulpStatus(status),
                [OptimalObjective(problem.objectives[0].name, pulp.value(pulp_problem.objective))],
                optimal_variables,
                dual_values,
            )

        else:
            status = problem.solve(solver)

            dual_values = [cons.pi for cons in problem.constraints.values()]
            if all(dv is None for dv in dual_values):
                dual_values = None

            return MIPSolution(
                PulpStatus(status),
                [OptimalObjective(problem.objective.name, pulp.value(problem.objective))],
                [OptimalVariable(var.name, pulp.value(var)) for var in problem.variables()],
                dual_values,
            )


class CplexSolver(object):
    """Wrapper around pulp.CPLEX_PY to integrate with GALINI."""
    def __init__(self, logger, config):
        self._inner = pulp.CPLEX_PY()
        self._logger = logger
        self._config = config.cplex

    def available(self):
        return self._inner.available()

    def actualSolve(self, lp):
        import cplex
        # Same as CPLEX_PY.actualSolve, but overrides log settings
        # and fixes some bugs
        self._inner.buildSolverModel(lp)

        self._setup_logs()

        model = lp.solverModel
        # set problem as mip if all continuous variables
        is_mip = True
        for var_type in model.variables.get_types():
            if var_type != 'C':
                is_mip = False
                break

        if is_mip:
            model.set_problem_type(cplex.Cplex.problem_type.LP)

        self._inner.callSolver(lp)
        solutionStatus = self._inner.findSolutionValues(lp)
        for var in lp.variables():
            var.modified = False
        for constraint in lp.constraints.values():
            constraint.modified = False

        # because of a bug in pulp, need to assign dual values here
        try:
            if model.get_problem_type() == cplex.Cplex.problem_type.LP:
                cons_name = [cons for cons in lp.constraints]
                constraintpivalues = dict(zip(cons_name,
                                              lp.solverModel.solution.get_dual_values(cons_name)))
                lp.assignConsPi(constraintpivalues)
        except cplex.exceptions.CplexSolverError:
            pass
        return solutionStatus

    def _setup_logs(self):
        if not self.available():
            return
        model = self._inner.solverModel
        model.set_warning_stream(_CplexLoggerAdapter(self._logger, WARNING))
        model.set_error_stream(_CplexLoggerAdapter(self._logger, ERROR))
        model.set_log_stream(_CplexLoggerAdapter(self._logger, INFO))
        model.set_results_stream(_CplexLoggerAdapter(self._logger, INFO))

        for key, value in self._config.items():
            # Parameters are specified as a path (mip.tolerances.mipgap)
            # access one attribute at the time.
            if key == 'timelimit':
                # make sure we respect global timelimit
                value = min(value, seconds_left())
            attr = model.parameters
            for p in key.split('.'):
                attr = getattr(attr, p)
            attr.set(value)


def _solver(logger, config):
    cplex = CplexSolver(logger, config)
    if cplex.available():
        return cplex
    return pulp.PULP_CBC_CMD()


def _dag_to_pulp(problem):
    assert len(problem.objectives) == 1
    objective = problem.objectives[0]
    assert objective.sense == Sense.MINIMIZE

    lp = pulp.LpProblem(problem.name, pulp.LpMinimize)

    variables = [_variable_to_pulp(problem, var) for var in problem.variables]

    lp += _linear_expression_to_pulp(variables, objective.root_expr)

    for constraint in problem.constraints:
        expr = _linear_expression_to_pulp(variables, constraint.root_expr)
        if constraint.lower_bound is None:
            lp += expr <= constraint.upper_bound
        elif constraint.upper_bound is None:
            lp += expr >= constraint.lower_bound
        else:
            lp += constraint.lower_bound <= expr
            lp += expr <= constraint.upper_bound
    return lp, variables


def _variable_to_pulp(problem, variable):
    view = problem.variable_view(variable)
    lower_bound = view.lower_bound()
    upper_bound = view.upper_bound()
    domain = pulp.LpContinuous
    if view.domain == Domain.INTEGER:
        domain = pulp.LpInteger
    elif view.domain == Domain.BINARY:
        domain = pulp.LpBinary
    return pulp.LpVariable(
        variable.name,
        lower_bound,
        upper_bound,
        domain,
    )


def _linear_expression_to_pulp(variables, expr):
    if expr.expression_type == ExpressionType.Variable:
        return variables[expr.idx]
    result = 0.0
    stack = [expr]
    visited = set()
    while len(stack) > 0:
        expr = stack.pop()
        visited.add(expr.uid)
        if expr.expression_type == ExpressionType.Sum:
            for child in expr.children:
                if child.uid not in visited:
                    stack.append(child)
        elif expr.expression_type == ExpressionType.Linear:
            result += expr.constant_term
            for child in expr.children:
                result += expr.coefficient(child) * variables[child.idx]
        else:
            assert expr.expression_type == ExpressionType.Variable
    return result


class _CplexLoggerAdapter(object):
    def __init__(self, logger, level):
        self._logger = logger
        self._level = level

    def write(self, msg):
        self._logger.log(self._level, msg)

    def flush(self):
        pass
