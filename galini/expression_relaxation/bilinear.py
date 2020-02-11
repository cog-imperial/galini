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
import numpy as np
from suspect.interval import Interval
from suspect.expression import ExpressionType
from galini.core import (
    QuadraticExpression,
    LinearExpression,
    SumExpression,
    Constraint,
    Variable,
    Domain,
    BilinearTermReference,
)
from galini.pyomo.postprocess import PROBLEM_BILINEAR_AUX_VAR_META
from galini.expression_relaxation.expression_relaxation import (
    ExpressionRelaxation,
    ExpressionRelaxationResult,
)

# Dictionary of (var1, var2) -> aux
BILINEAR_AUX_VAR_META = 'bilinear_aux_variables'


class McCormickExpressionRelaxation(ExpressionRelaxation):
    """Underestimate Quadratic expressions using McCormick envelope."""
    def __init__(self, linear=True, disable_midpoint=False):
        self.linear = linear
        self.disable_midpoint = disable_midpoint

    def can_relax(self, problem, expr, ctx):
        return expr.expression_type == ExpressionType.Quadratic

    def relax(self, problem, expr, ctx, **kwargs):
        assert expr.expression_type == ExpressionType.Quadratic

        if ctx.metadata.get(BILINEAR_AUX_VAR_META, None) is None:
            ctx.metadata[BILINEAR_AUX_VAR_META] = dict()

        squares = []
        variables = []
        constraints = []
        for term in expr.terms:
            if term.var1 != term.var2 or self.linear:
                aux_var_linear, aux_var_constraints = \
                    self._underestimate_bilinear_term(problem, term, ctx)
                if aux_var_linear is None:
                    return None
                variables.append(aux_var_linear)
                constraints.extend(aux_var_constraints)
            else:
                squares.append((term.coefficient, term.var1))

        if not squares:
            new_linear_expr = LinearExpression(variables)
            return ExpressionRelaxationResult(new_linear_expr, constraints)

        # Squares + (optional) linear expression
        square_coefficients = [c for c, _ in squares]
        square_variables = [v for _, v in squares]
        quadratic_expr = QuadraticExpression(
            square_variables,
            square_variables,
            square_coefficients,
        )
        if not variables:
            return ExpressionRelaxationResult(quadratic_expr, constraints)

        new_linear_expr = LinearExpression(variables)
        return ExpressionRelaxationResult(
            SumExpression([quadratic_expr, new_linear_expr]),
            constraints,
        )

    def _get_bilinear_aux_var(self, problem, ctx, xy_tuple):
        bilinear_aux_variables = \
            ctx.metadata.get(BILINEAR_AUX_VAR_META, dict())
        aux = bilinear_aux_variables.get(xy_tuple, None)

        if aux is not None:
            return aux

        problem_bilinear_aux_variables = \
            problem.root.metadata.get(PROBLEM_BILINEAR_AUX_VAR_META, dict())

        aux = problem_bilinear_aux_variables.get(xy_tuple, None)
        if aux is None:
            return None

        return problem.variable(aux)

    def _underestimate_bilinear_term(self, problem, term, ctx):
        bilinear_aux_vars = ctx.metadata[BILINEAR_AUX_VAR_META]
        x_expr = term.var1
        y_expr = term.var2

        xy_tuple = self._bilinear_tuple(x_expr, y_expr)
        w = self._get_bilinear_aux_var(problem, ctx, xy_tuple)

        if w is None:
            x_l = problem.lower_bound(x_expr)
            x_u = problem.upper_bound(x_expr)

            y_l = problem.lower_bound(y_expr)
            y_u = problem.upper_bound(y_expr)

            if term.var1 == term.var2:
                assert np.isclose(x_l, y_l) and np.isclose(x_u, y_u)
                w_bounds = Interval(x_l, x_u) ** 2
            else:
                w_bounds = Interval(x_l, x_u) * Interval(y_l, y_u)

            w = Variable(
                self._format_aux_name(term.var1, term.var2),
                w_bounds.lower_bound,
                w_bounds.upper_bound,
                Domain.REAL,
            )

            reference = BilinearTermReference(x_expr, y_expr)

            w.reference = reference
            bilinear_aux_vars[xy_tuple] = w
            constraints = self._generate_envelope_constraints(problem, term, w)
        else:
            constraints = []

        new_expr = LinearExpression([w], [term.coefficient], 0.0)

        return new_expr, constraints

    def _generate_envelope_constraints(self, problem, term, w):
        x_expr = term.var1
        y_expr = term.var2

        x = problem.variable_view(x_expr)
        y = problem.variable_view(y_expr)

        x_l = x.lower_bound()
        x_u = x.upper_bound()

        y_l = y.lower_bound()
        y_u = y.upper_bound()

        #  y     x     w   const
        #  x^L   y^L  -1  -x^L y^L <= 0
        #  x^U   y^U  -1  -x^U y^U <= 0
        # -x^U  -y^L  +1  +x^U y^L <= 0
        # -x^L  -y^U  +1  +x^L y^U <= 0

        constraints = []

        if term.var1 == term.var2:
            if not _is_inf(x_l):
                lower_bound_0 = Constraint(
                    self._format_constraint_name(w, 'lb_0'),
                    LinearExpression([x_expr, w], [2.0*x_l, -1], -x_l*x_l),
                    None,
                    0.0,
                )
                constraints.append(lower_bound_0)

            if not _is_inf(x_u):
                lower_bound_1 = Constraint(
                    self._format_constraint_name(w, 'lb_1'),
                    LinearExpression([x_expr, w], [2.0*x_u, -1], -x_u*x_u),
                    None,
                    0.0,
                )
                constraints.append(lower_bound_1)

            if not _is_inf(x_u) and not _is_inf(x_l):
                upper_bound_0 = Constraint(
                    self._format_constraint_name(w, 'ub_0'),
                    LinearExpression([x_expr, w], [-(x_u + x_l), 1], x_u*x_l),
                    None,
                    0.0,
                )
                constraints.append(upper_bound_0)

                # x_m = x_l + 0.5 * (x_u - x_l)
                x_m = x_l + (x_u - x_l) / 2.0
                lower_bound_2 = Constraint(
                    self._format_constraint_name(w, 'lb_2'),
                    LinearExpression([x_expr, w], [2.0*x_m, -1], -x_m*x_m),
                    None,
                    0.0,
                )
                if not self.disable_midpoint:
                    constraints.append(lower_bound_2)

        else:
            if not _is_inf(x_l) and not _is_inf(y_l):
                lower_bound_0 = Constraint(
                    self._format_constraint_name(w, 'lb_0'),
                    LinearExpression([y_expr, x_expr, w], [x_l, y_l, -1], -x_l*y_l),
                    None,
                    0.0,
                )
                constraints.append(lower_bound_0)

            if not _is_inf(x_u) and not _is_inf(y_u):
                lower_bound_1 = Constraint(
                    self._format_constraint_name(w, 'lb_1'),
                    LinearExpression([y_expr, x_expr, w], [x_u, y_u, -1], -x_u*y_u),
                    None,
                    0.0,
                )
                constraints.append(lower_bound_1)

            if not _is_inf(x_u) and not _is_inf(y_l):
                upper_bound_0 = Constraint(
                    self._format_constraint_name(w, 'ub_0'),
                    LinearExpression([y_expr, x_expr, w], [-x_u, -y_l, 1], x_u*y_l),
                    None,
                    0.0,
                )
                constraints.append(upper_bound_0)

            if not _is_inf(x_l) and not _is_inf(y_u):
                upper_bound_1 = Constraint(
                    self._format_constraint_name(w, 'ub_1'),
                    LinearExpression([y_expr, x_expr, w], [-x_l, -y_u, 1], x_l*y_u),
                    None,
                    0.0,
                )
                constraints.append(upper_bound_1)

        return constraints

    def _format_aux_name(self, x, y):
        return '_aux_bilinear_{}_{}'.format(x.name, y.name)

    def _format_constraint_name(self, v, suffix):
        return '_mccormick_{}_{}'.format(v.name, suffix)

    def _bilinear_tuple(self, x, y):
        x_uid = x.idx
        y_uid = y.idx
        return min(x_uid, y_uid), max(x_uid, y_uid)


def _is_inf(n):
    return n is None or np.isinf(n)
