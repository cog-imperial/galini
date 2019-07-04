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

"""Replace nonlinear expressions with auxiliary variables."""

from suspect.expression import ExpressionType
from galini.core import (
    LinearExpression,
    SumExpression,
    Constraint,
    Domain,
    ExpressionReference,
)
from galini.transformation import Transformation, TransformationResult


class ReplaceNonlinearTransformation(Transformation):
    """Replace Nonlinear expressions with auxiliary variables."""
    def __init__(self, source, target):
        super().__init__(source, target)
        self._memo = {}

    def apply(self, expr, ctx):
        assert expr.problem is None or expr.problem == self.source
        if expr.expression_type != ExpressionType.Sum:
            return TransformationResult(expr, [])

        # Rewrite expression to be a linear expression containing
        # sum of original variables and auxiliary variables for nonlinear
        # expressions
        new_variables = []
        new_constraints = []
        original_linear_expressions = []
        for child in expr.children:
            if child.expression_type == ExpressionType.Linear:
                original_linear_expressions.append(child)
            else:
                w, constraint = self._build_constraint(child, ctx)
                new_variables.append(w)
                if constraint is not None:
                    new_constraints.append(constraint)

        linear_expr = self._build_linear_expression(original_linear_expressions, new_variables)
        return TransformationResult(linear_expr, new_constraints)

    def _build_constraint(self, expr, ctx):
        if expr.uid in self._memo:
            return self._memo[expr.uid], None
        expr_bounds = ctx.bounds(expr)
        reference = ExpressionReference(expr)
        # Use bounds
        w = Variable(
            self._format_aux_name(expr),
            expr_bounds.lower_bound,
            expr_bounds.upper_bound,
            Domain.REAL,
        )
        w.reference = reference
        new_expr = SumExpression([
            LinearExpression([w], [-1.0], 0.0),
            expr
        ])
        new_constraint = Constraint(
            self._format_constraint_name(w),
            new_expr,
            0.0,
            0.0,
        )
        self._memo[expr.uid] = w
        return w, new_constraint

    def _build_linear_expression(self, original_linear_expressions, new_variables):
        linear_vars = {}
        constant_term = 0.0
        for linear_expr in original_linear_expressions:
            for child in linear_expr.children:
                if child.uid not in linear_vars:
                    coef = 0.0
                else:
                    coef, _ = linear_vars[child.uid]

                coef += linear_expr.coefficient(child)
                linear_vars[child.uid] = (coef, child)

            constant_term += linear_expr.constant_term

        for var in new_variables:
            if var.uid in linear_vars:
                coef, _ = linear_vars[var.uid]
            else:
                coef = 0.0
            coef += 1.0
            linear_vars[var.uid] = (coef, var)

        new_children = []
        new_coefficients = []
        for _, (coef, var) in linear_vars.items():
            new_children.append(var)
            new_coefficients.append(coef)

        return LinearExpression(
            new_children,
            new_coefficients,
            constant_term,
        )

    def _format_aux_name(self, expr):
        return '_aux_{}'.format(expr.uid)

    def _format_constraint_name(self, w):
        return '_nonlinear_{}'.format(w.name)
