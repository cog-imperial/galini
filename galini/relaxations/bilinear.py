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

"""Bilinear relaxation using McCormick."""
from suspect.expression import ExpressionType
from galini.relaxations.relaxation import Relaxation, RelaxationResult, NewVariable


class McCormickRelaxation(Relaxation):
    """Relax bilinear terms using McCormick envelope."""
    def can_relax(self, expr, ctx):
        poly = ctx.polynomial(expr)
        return poly.is_quadratic()

    def relax(self, expr, ctx):
        assert expr.expression_type == ExpressionType.Product
        x, y, c = self._get_variables(expr)

        w = NewVariable()

        x_l = x.lower_bound()
        x_u = x.upper_bound()

        y_l = y.lower_bound()
        y_u = y.upper_bound()

    def _get_variables(self, expr):
        x = expr.children[0]
        y = expr.children[1]

        if x.expression_type == ExpressionType.Linear:
            assert len(x.children) == 1
            assert y.expression_type == ExpressionType.Variable
            c = x.coefficients[0]
            x = x.children[0]
            return x, y, c

        if y.expression_type == ExpressionType.Linear:
            assert len(y.children) == 1
            assert x.expression_type == ExpressionType.Variable
            c = y.coefficients[0]
            y = y.children[0]
            return x, y, c

        assert x.expression_type == ExpressionType.Variable
        assert y.expression_type == ExpressionType.Variable
        return x, y, 1.0
