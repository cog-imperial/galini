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

"""Underestimator for Convex functions."""
import math
import numpy as np
from suspect.expression import ExpressionType
from galini.core import QuadraticExpression
from galini.underestimators.underestimator import Underestimator, UnderestimatorResult


class ConvexUnderestimator(Underestimator):
    """Underestimator of convex functions."""
    def can_underestimate(self, problem, expr, ctx):
        cvx = ctx.convexity(expr)
        # special case for quadratic, where square terms
        # are convex
        return cvx.is_convex() or expr.expression_type == ExpressionType.Quadratic

    def underestimate(self, problem, expr, ctx, **kwargs):
        cvx = ctx.convexity(expr)
        if cvx.is_convex():
            return UnderestimatorResult(expr)

        assert expr.expression_type == ExpressionType.Quadratic
        # build quadratic of only the squares
        variables = []
        coefficients = []
        for term in expr.terms:
            if term.var1 == term.var2:
                variables.append(term.var1)
                coefficients.append(term.coefficient)
        new_expr = QuadraticExpression(variables, variables, coefficients)
        return UnderestimatorResult(new_expr)
