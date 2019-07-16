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

import sys
from galini.core import Sense
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


def expr_to_str(expr):
    et = expr.expression_type
    children_str = [expr_to_str(ch) for ch in expr.children]

    if et == ExpressionType.Sum:
        return ' + '.join(children_str)
    elif et == ExpressionType.Variable:
        return expr.name
    elif et == ExpressionType.Constant:
        return str(expr.value)
    elif et == ExpressionType.Division:
        return '({}) / ({})'.format(children_str[0], children_str[1])
    elif et == ExpressionType.Product:
        return '({}) * ({})'.format(children_str[0], children_str[1])
    elif et == ExpressionType.Linear:
        return (' + '.join(['{} {}'.format(expr.coefficient(ch), ch.name) for ch in expr.children])
                + ' + ' + str(expr.constant_term))
    elif et == ExpressionType.Power:
        return 'pow({}, {})'.format(children_str[0], children_str[1])
    elif et == ExpressionType.Negation:
        return '-({})'.format(children_str[0])
    elif et == ExpressionType.Quadratic:
        return ' + '.join(['{} {} {}'.format(t.coefficient, t.var1.name, t.var2.name) for t in expr.terms])
    elif et == ExpressionType.UnaryFunction:
        return _FUNC_TYPE_TO_CLS[expr.func_type] + '(' + children_str[0] + ')'

    else:
        raise ValueError('Unhandled expression_type {}'.format(et))
