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

"""Bilinear underestimator using McCormick."""
import numpy as np
from suspect.expression import ExpressionType
from galini.core import LinearExpression, Variable, Constraint, Domain
from galini.underestimators.underestimator import Underestimator, UnderestimatorResult


class McCormickUnderestimator(Underestimator):
    """Underestimate bilinear terms using McCormick envelope."""
    def can_underestimate(self, problem, expr, ctx):
        poly = ctx.polynomial(expr)
        return poly.is_quadratic() and expr.expression_type == ExpressionType.Product

    def underestimate(self, problem, expr, ctx):
        assert expr.expression_type == ExpressionType.Product
        x_expr, y_expr, c = self._get_variables(expr)

        x = problem.variable_view(x_expr)
        y = problem.variable_view(y_expr)

        x_l = x.lower_bound()
        x_u = x.upper_bound()

        y_l = y.lower_bound()
        y_u = y.upper_bound()

        if x_l is None or x_u is None or y_l is None or y_u is None:
            return None

        w = Variable(self._format_aux_name(x_expr, y_expr), None, None, Domain.REAL)

        #  y     x     w   const
        #  x^L   y^L  -1  -x^L y^L <= 0
        #  x^U   y^U  -1  -x^U y^U <= 0
        # -x^U  -y^L  +1  +x^U y^L <= 0
        # -x^L  -y^U  +1  +x^L y^U <= 0

        upper_bound_0 = Constraint(
            self._format_constraint_name(w, 'ub_0'),
            LinearExpression([y_expr, x_expr, w], [x_l, y_l, -1], -x_l*y_l),
            None,
            0.0,
        )
        upper_bound_1 = Constraint(
            self._format_constraint_name(w, 'ub_1'),
            LinearExpression([y_expr, x_expr, w], [x_u, y_u, -1], -x_u*y_u),
            None,
            0.0,
        )
        lower_bound_0 = Constraint(
            self._format_constraint_name(w, 'lb_0'),
            LinearExpression([y_expr, x_expr, w], [-x_u, -y_l, 1], x_u*y_l),
            0.0,
            None,
        )
        lower_bound_1 = Constraint(
            self._format_constraint_name(w, 'lb_1'),
            LinearExpression([y_expr, x_expr, w], [-x_l, -y_u, 1], x_l*y_u),
            0.0,
            None,
        )

        if isinstance(c, float) and not np.isclose(c, 1.0):
            new_expr = LinearExpression([w], [c], 0.0)
        else:
            new_expr = w
        return UnderestimatorResult(
            new_expr,
            [upper_bound_0, upper_bound_1, lower_bound_0, lower_bound_1]
        )

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

        if x.expression_type == ExpressionType.Product:
            assert y.expression_type == ExpressionType.Constant
            x_ = x.children[1]
            y_ = x.children[0]
            c = y.value
            return x_, y_, c

        assert x.expression_type == ExpressionType.Variable
        assert y.expression_type == ExpressionType.Variable
        return x, y, 1.0

    def _format_aux_name(self, x, y):
        return '_aux_bilinear_{}_{}'.format(x.name, y.name)

    def _format_constraint_name(self, v, suffix):
        return '_mccormick_{}_{}'.format(v.name, suffix)
