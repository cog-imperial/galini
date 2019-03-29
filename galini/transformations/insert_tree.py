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

"""Insert expression tree in problem, keeping a memo to avoid duplication."""
from suspect.expression import ExpressionType, UnaryFunctionType
import galini.core as core


def insert_tree(source, target, expr, memo):
    """Insert `expr` in `source`, keeping node duplication at minimum."""
    assert memo is not None
    new_expr = _duplicate_expr(source, target, expr, memo)
    target.insert_tree(new_expr)
    return new_expr


def _duplicate_expr(source, target, expr, memo):
    if expr.uid in memo:
        return memo[expr.uid]

    if expr.expression_type == ExpressionType.Variable:
        if expr.problem == source:
            return target.variable(expr.idx)
        else:
            if isinstance(expr, core.Variable):
                new_var = target.add_variable(
                    expr.name,
                    expr.lower_bound,
                    expr.upper_bound,
                    expr.domain,
                )
            else:
                assert isinstance(expr, core.AuxiliaryVariable)
                new_var = target.add_aux_variable(
                    expr.name,
                    expr.lower_bound,
                    expr.upper_bound,
                    expr.domain,
                    expr.reference,
                )

        memo[expr.uid] = new_var
        return new_var

    children = [_duplicate_expr(source, target, child, memo) for child in expr.children]
    new_expr = _make_expr(expr, children)
    memo[expr.uid] = new_expr
    return new_expr


_EXPR_TYPE_TO_CLS = {
    ExpressionType.Product: core.ProductExpression,
    ExpressionType.Division: core.DivisionExpression,
    ExpressionType.Sum: core.SumExpression,
    ExpressionType.Power: core.PowExpression,
    ExpressionType.Negation: core.NegationExpression,
}


_FUNC_TYPE_TO_CLS = {
    UnaryFunctionType.Abs: core.AbsExpression,
    UnaryFunctionType.Sqrt: core.SqrtExpression,
    UnaryFunctionType.Exp: core.ExpExpression,
    UnaryFunctionType.Log: core.LogExpression,
    UnaryFunctionType.Sin: core.SinExpression,
    UnaryFunctionType.Cos: core.CosExpression,
    UnaryFunctionType.Tan: core.TanExpression,
    UnaryFunctionType.Asin: core.AsinExpression,
    UnaryFunctionType.Acos: core.AcosExpression,
    UnaryFunctionType.Atan: core.AtanExpression,
}


def _make_expr(expr, children):
    type_ = expr.expression_type
    if type_ == ExpressionType.Linear:
        coefficients = [expr.coefficient(v) for v in expr.children]
        return core.LinearExpression(children, coefficients, expr.constant_term)
    elif type_ == ExpressionType.Quadratic:
        child_by_index = dict([(ch.idx, ch) for ch in children])
        terms = expr.terms
        coefficients = [t.coefficient for t in terms]
        vars1 = [child_by_index[t.var1.idx] for t in terms]
        vars2 = [child_by_index[t.var2.idx] for t in terms]
        return core.QuadraticExpression(vars1, vars2, coefficients)
    elif type_ == ExpressionType.Constant:
        return core.Constant(expr.value)
    elif type_ == ExpressionType.UnaryFunction:
        func_type = expr.func_type
        cls = _FUNC_TYPE_TO_CLS[func_type]
        return cls(children)
    cls = _EXPR_TYPE_TO_CLS[type_]
    return cls(children)
