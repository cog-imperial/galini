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

"""McCormick envelops of bilinear terms."""
import numpy as np
from suspect.interval import Interval
from suspect.expression import ExpressionType
from galini.core import (
    LinearExpression,
    Constraint,
    Domain,
    BilinearTermReference,
)
from galini.transformation import Transformation, TransformationResult


class McCormickTransformation(Transformation):
    """McCormick envelopes for bilinear terms."""
    def __init__(self, source_problem, target_problem):
        super().__init__(source_problem, target_problem)
        self._aux_variables = {}

    def apply(self, expr, ctx):
        assert expr.problem is None or expr.problem == self.source
        if expr.expression_type != ExpressionType.Quadratic:
            return None
        variables = []
        constraints = []
        for term in expr.terms:
            aux_var_linear, aux_var_constraints = \
                self._underestimate_bilinear_term(self.source, term, ctx)
            if aux_var_linear is None:
                return None
            variables.append(aux_var_linear)
            constraints.extend(aux_var_constraints)

        new_linear_expr = LinearExpression(variables)
        return TransformationResult(new_linear_expr, constraints)

    def _underestimate_bilinear_term(self, problem, term, _ctx):
        x_expr = term.var1
        y_expr = term.var2

        xy_tuple = self._bilinear_tuple(x_expr, y_expr)
        if xy_tuple in self._aux_variables:
            w = self._aux_variables[xy_tuple]
            new_expr = LinearExpression([w], [term.coefficient], 0.0)
            return new_expr, []

        x = problem.variable_view(term.var1)
        y = problem.variable_view(term.var2)

        x_l = x.lower_bound()
        x_u = x.upper_bound()

        y_l = y.lower_bound()
        y_u = y.upper_bound()

        assert not _is_inf(x_l)
        assert not _is_inf(x_u)
        assert not _is_inf(y_l)
        assert not _is_inf(y_u)

        if term.var1 == term.var2:
            assert np.isclose(x_l, y_l) and np.isclose(x_u, y_u)
            w_bounds = Interval(x_l, x_u) ** 2
        else:
            w_bounds = Interval(x_l, x_u) * Interval(y_l, y_u)

        reference = BilinearTermReference(term.var1, term.var2)
        w = Variable(
            self._format_aux_name(term.var1, term.var2),
            w_bounds.lower_bound,
            w_bounds.upper_bound,
            Domain.REAL,
        )
        w.reference = reference
        self._aux_variables[xy_tuple] = w

        #  y     x     w   const
        #  x^L   y^L  -1  -x^L y^L <= 0
        #  x^U   y^U  -1  -x^U y^U <= 0
        # -x^U  -y^L  +1  +x^U y^L <= 0
        # -x^L  -y^U  +1  +x^L y^U <= 0

        new_expr = LinearExpression([w], [term.coefficient], 0.0)

        if term.var1 == term.var2:
            new_constraints = self._transform_square(w, x_expr, y_expr, x_l, x_u, y_l, y_u)
        else:
            new_constraints = self._transform_bilinear(w, x_expr, y_expr, x_l, x_u, y_l, y_u)

        return new_expr, new_constraints

    def _transform_square(self, w, x_expr, _y_expr, x_l, x_u, _y_l, _y_u):
        upper_bound_0 = Constraint(
            self._format_constraint_name(w, 'ub_0'),
            LinearExpression([x_expr, w], [2.0*x_l, -1], -x_l*x_l),
            None,
            0.0,
        )
        upper_bound_1 = Constraint(
            self._format_constraint_name(w, 'ub_1'),
            LinearExpression([x_expr, w], [2.0*x_u, -1], -x_u*x_u),
            None,
            0.0,
        )
        lower_bound_0 = Constraint(
            self._format_constraint_name(w, 'lb_0'),
            LinearExpression([x_expr, w], [-(x_u + x_l), 1], x_u*x_l),
            None,
            0.0,
        )

        return [upper_bound_0, upper_bound_1, lower_bound_0]

    def _transform_bilinear(self, w, x_expr, y_expr, x_l, x_u, y_l, y_u):
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

        return [upper_bound_0, upper_bound_1, lower_bound_0, lower_bound_1]

    def _format_aux_name(self, x, y):
        return '_aux_bilinear_{}_{}'.format(x.name, y.name)

    def _format_constraint_name(self, v, suffix):
        return '_mccormick_{}_{}'.format(v.name, suffix)

    def _bilinear_tuple(self, x, y):
        x_uid = x.uid
        y_uid = y.uid
        return min(x_uid, y_uid), max(x_uid, y_uid)

def _is_inf(n):
    return n is None or np.isinf(n)
