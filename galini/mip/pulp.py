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
"""Utilities to interface GALINI with pulp."""
import pulp
from suspect.expression import ExpressionType
from galini.core import Sense, Domain
from galini.mip.cplex import CplexSolver


def mip_solver(logger, run_id, galini):
    """Get a pulp mip solver."""
    cplex = CplexSolver(logger, run_id, galini)
    if cplex.available():
        logger.debug(run_id, 'Using CPLEX as MILP solver')
        return cplex
    logger.debug(run_id, 'Using CBC as MILP solver')
    return pulp.PULP_CBC_CMD()


def dag_to_pulp(problem):
    """Convert GALINI DAG to pulp model."""
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
    while stack:
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
