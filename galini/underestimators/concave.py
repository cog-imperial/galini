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

"""Underestimator for Univariate Concave functions."""
import math
import numpy as np
from suspect.expression import ExpressionType, UnaryFunctionType
from galini.core import LinearExpression, Variable, Constraint, Domain
from galini.underestimators.underestimator import Underestimator, UnderestimatorResult


class UnivariateConcaveUnderestimator(Underestimator):
    """Underestimator of univariate concave functions."""
    def can_underestimate(self, problem, expr, ctx):
        cvx = ctx.convexity(expr)
        return cvx.is_concave() and self.is_univariate(expr)

    def underestimate(self, problem, expr, ctx):
        var = self._collect_variable(expr)
        if var is None:
            raise RuntimeError('Not an unary expression')
        var_view = problem.variable_view(var)
        lower_bound = var_view.lower_bound()
        upper_bound = var_view.upper_bound()

        if lower_bound is None or upper_bound is None:
            return None

        ut_xl = _eval_univariate_expression_at_point(expr, lower_bound)
        ut_xu = _eval_univariate_expression_at_point(expr, upper_bound)

        # the underestimator is
        #   ut_xl + (x - xl) * c
        # where c is
        #   (ut_xu - ut_xl) / (xu - xl)
        # can be rewritten as the linear expression with coefficient c
        # and constant term
        #   -(ut_xl * c * xl)
        c = (ut_xu - ut_xl) / (upper_bound - lower_bound)
        const = -c * ut_xl * lower_bound

        new_expr = LinearExpression([var], [c], const)
        return UnderestimatorResult(new_expr)

    def is_univariate(self, expr):
        try:
            var = self._collect_variable(expr)
            return var is not None
        except:
            return False

    def _collect_variable(self, expr):
        # we are drilling down a DAG, so we need to avoid visiting the
        # same nodes twice (think of diamond situation).
        seen = set()
        stack = [expr]
        variable = None
        while len(stack) > 0:
            current = stack.pop()
            et = current.expression_type
            if et == ExpressionType.Variable:
                if variable is not None:
                    raise RuntimeError('Not an unary expression')
                variable = current
            seen.add(current.idx)
            for ch in current.children:
                if ch.idx not in seen:
                    stack.append(ch)
        return variable


def _eval_univariate_expression_at_point(expr, value):
    et = expr.expression_type
    if et == ExpressionType.UnaryFunction:
        return _eval_univariate_function(expr, value)
    elif et == ExpressionType.Variable:
        return value
    elif et == ExpressionType.Constant:
        return expr.value
    elif et == ExpressionType.Division:
        num, den = et.children
        return (
            _eval_univariate_expression_at_point(num, value) /
            _eval_univariate_expression_at_point(den, value)
        )
    elif et == ExpressionType.Product:
        lhs, rhs = et.children
        return (
            _eval_univariate_expression_at_point(lhs, value) *
            _eval_univariate_expression_at_point(rhs, value)
        )
    elif et == ExpressionType.Linear:
        assert len(expr.children) == 1
        coef = expr.coefficients[0]
        return value * coef + expr.constant_term
    elif et == ExpressionType.Sum:
        return sum(_eval_univariate_expression_at_point(ch, value) for ch in expr.children)
    elif et == ExpressionType.Power:
        base, expo = expr.children
        return (
            _eval_univariate_expression_at_point(base, value) **
            _eval_univariate_expression_at_point(expo, value)
        )
    elif et == ExpressionType.Negation:
        return -_eval_univariate_expression_at_point(expr.children[0], value)
    else:
        raise RuntimeError('Invalid ExpressionType {}'.format(et))


def _eval_univariate_function(expr, value):
    child = expr.children[0]
    child_value = _eval_univariate_expression_at_point(child, value)

    func_type_to_func = {
        UnaryFunctionType.Abs: abs,
        UnaryFunctionType.Sqrt: math.sqrt,
        UnaryFunctionType.Exp: math.exp,
        UnaryFunctionType.Log: math.log,
        UnaryFunctionType.Sin: math.sin,
        UnaryFunctionType.Cos: math.cos,
        UnaryFunctionType.Tan: math.tan,
        UnaryFunctionType.Asin: math.asin,
        UnaryFunctionType.Acos: math.acos,
        UnaryFunctionType.Atan: math.atan,
    }
    func = func_type_to_func.get(expr.func_type)

    if func is None:
        raise RuntimeError('Invalid UnaryFunctionType {}'.format(expr.func_type))

    return func(child_value)
