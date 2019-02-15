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
from suspect.expression import ExpressionType


def print_problem(problem, out=None):
    """Outputs the problem to console."""
    if out is None:
        out = sys.stdout
    for objective in problem.objectives:
        sense_name = {
            Sense.MINIMIZE: 'min',
            Sense.MAXIMIZE: 'max',
        }
        out.write(sense_name[objective.sense] + ' ')
        out.write(print_expr(objective.root_expr))
        out.write('\n')

    for constraint in problem.constraints:
        out.write('{}:\t'.format(constraint.name))
        out.write('{} <= '.format(constraint.lower_bound))
        out.write(print_expr(constraint.root_expr))
        out.write(' <= {}'.format(constraint.upper_bound))
        out.write('\n')
    out.write('\n')


def print_expr(expr):
    et = expr.expression_type
    if et == ExpressionType.Sum:
        return ' + '.join([print_expr(ch) for ch in expr.children])
    elif et == ExpressionType.Variable:
        return expr.name
    elif et == ExpressionType.Linear:
        return (' + '.join(['{} {}'.format(expr.coefficient(ch), ch.name) for ch in expr.children])
                + ' + ' + str(expr.constant_term))
    else:
        raise ValueError('Unhandled expression_type {}'.format(et))
