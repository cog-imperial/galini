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

"""GALINI utility functions."""
import sys

from suspect.expression import ExpressionType, UnaryFunctionType


def print_problem(problem, out=None):
    """Outputs the problem to console."""
    if out is None:
        out = sys.stdout
    objective = problem.objective
    out.write('min ')
    out.write(expr_to_str(objective.root_expr))
    out.write('\n')

    for constraint in problem.constraints:
        out.write('{}:\t'.format(constraint.name))
        out.write('{} <= '.format(constraint.lower_bound))
        out.write(expr_to_str(constraint.root_expr))
        out.write(' <= {}'.format(constraint.upper_bound))
        out.write('\n')
    out.write('\n')


def log_problem(logger, run_id, level, problem, title=None):
    """Write problem to `logger` with the given level."""
    if logger.level > level:
        return

    if title:
        logger.log(run_id, level, 'Problem: {}'.format(title))

    logger.log(run_id, level, 'Variables:')
    for v in problem.variables:
        vv = problem.variable_view(v)
        logger.log(
            run_id, level,
            '\t{}\t[{}, {}]\t{}',
            v.name,
            vv.lower_bound(),
            vv.upper_bound(),
            vv.domain,
        )

    logger.log(run_id, level, 'Objective:')
    logger.log(
        run_id, level,
        'O {}\n\t{}',
        problem.objective.name,
        expr_to_str(problem.objective.root_expr)
    )

    logger.log(run_id, level, 'Constraints:')
    for constraint in problem.constraints:
        logger.log(
            run_id, level,
            'C {}\n\t{} <= {} <= {}',
            constraint.name,
            constraint.lower_bound,
            expr_to_str(constraint.root_expr),
            constraint.upper_bound,
        )


_FUNC_TYPE_TO_CLS = {
    UnaryFunctionType.Abs: 'abs',
    UnaryFunctionType.Sqrt: 'sqrt',
    UnaryFunctionType.Exp: 'exp',
    UnaryFunctionType.Log: 'log',
    UnaryFunctionType.Sin: 'sin',
    UnaryFunctionType.Cos: 'cos',
    UnaryFunctionType.Tan: 'tan',
    UnaryFunctionType.Asin: 'asin',
    UnaryFunctionType.Acos: 'acos',
    UnaryFunctionType.Atan: 'atan',
}


# pylint: disable=invalid-name
def expr_to_str(expr):
    """Convert expression to string for debugging."""
    et = expr.expression_type
    children_str = [expr_to_str(ch) for ch in expr.children]

    if et == ExpressionType.Sum:
        return ' + '.join(children_str)

    if et == ExpressionType.Variable:
        return expr.name

    if et == ExpressionType.Constant:
        return str(expr.value)

    if et == ExpressionType.Division:
        return '({}) / ({})'.format(children_str[0], children_str[1])

    if et == ExpressionType.Product:
        return '({}) * ({})'.format(children_str[0], children_str[1])

    if et == ExpressionType.Linear:
        var_with_coef = [
            '{} {}'.format(expr.coefficient(ch), ch.name)
            for ch in expr.children
        ]
        return (' + '.join(var_with_coef)
                + ' + ' + str(expr.constant_term))

    if et == ExpressionType.Power:
        return 'pow({}, {})'.format(children_str[0], children_str[1])

    if et == ExpressionType.Negation:
        return '-({})'.format(children_str[0])

    if et == ExpressionType.Quadratic:
        term_with_coef = [
            '{} {} {}'.format(t.coefficient, t.var1.name, t.var2.name)
            for t in expr.terms
        ]
        return ' + '.join(term_with_coef)

    if et == ExpressionType.UnaryFunction:
        return _FUNC_TYPE_TO_CLS[expr.func_type] + '(' + children_str[0] + ')'

    raise ValueError('Unhandled expression_type {}'.format(et))


def solution_numerical_value(solution, var_lb, var_ub):
    """Return the solution value if present, use bounds to compute if not."""
    value = solution.value
    if value is not None:
        return value

    if var_lb is not None and var_ub is not None:
        return var_lb + (var_ub / 2.0)

    if var_lb is None and var_ub is None:
        return 0.0

    if var_lb is None:
        return var_ub

    # var_ub is None
    return var_lb
