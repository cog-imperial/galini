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
    """Underestimate Quadratic expressions using McCormick envelope."""
    def can_underestimate(self, problem, expr, ctx):
        return expr.expression_type == ExpressionType.Quadratic

    def underestimate(self, problem, expr, ctx):
        assert expr.expression_type == ExpressionType.Quadratic
        if ctx.metadata.get('bilinear_aux_variables', None) is None:
            ctx.metadata['bilinear_aux_variables'] = {}
        variables = []
        constraints = []
        for term in expr.terms:
            aux_var_linear, aux_var_constraints = self._underestimate_bilinear_term(problem, term, ctx)
            if aux_var_linear is None:
                return None
            variables.append(aux_var_linear)
            constraints.extend(aux_var_constraints)

        new_linear_expr = LinearExpression(variables)
        return UnderestimatorResult(new_linear_expr, constraints)

    def _underestimate_bilinear_term(self, problem, term, ctx):
        bilinear_aux_vars = ctx.metadata['bilinear_aux_variables']
        x_expr = term.var1
        y_expr = term.var2

        xy_tuple = self._bilinear_tuple(x_expr, y_expr)
        if xy_tuple in bilinear_aux_vars:
            w = bilinear_aux_vars[xy_tuple]
            new_expr = LinearExpression([w], [term.coefficient], 0.0)
            return new_expr, []

        x = problem.variable_view(term.var1)
        y = problem.variable_view(term.var2)

        x_l = x.lower_bound()
        x_u = x.upper_bound()

        y_l = y.lower_bound()
        y_u = y.upper_bound()

        if x_l is None or x_u is None or y_l is None or y_u is None:
            return None, []

        w = Variable(self._format_aux_name(term.var1, term.var2), None, None, Domain.REAL)
        bilinear_aux_vars[xy_tuple] = w

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
            None,
            0.0,
        )
        lower_bound_1 = Constraint(
            self._format_constraint_name(w, 'lb_1'),
            LinearExpression([y_expr, x_expr, w], [-x_l, -y_u, 1], x_l*y_u),
            None,
            0.0,
        )

        new_expr = LinearExpression([w], [term.coefficient], 0.0)
        return new_expr, [upper_bound_0, upper_bound_1, lower_bound_0, lower_bound_1]

    def _format_aux_name(self, x, y):
        return '_aux_bilinear_{}_{}'.format(x.name, y.name)

    def _format_constraint_name(self, v, suffix):
        return '_mccormick_{}_{}'.format(v.name, suffix)

    def _bilinear_tuple(self, x, y):
        x_uid = x.uid
        y_uid = y.uid
        return min(x_uid, y_uid), max(x_uid, y_uid)
