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
"""Convert Pyomo problems to GALINI Problem."""
from numbers import Number
import warnings
import numpy as np
import pyomo.environ as aml
from pyomo.core.expr.visitor import ExpressionValueVisitor
from pyomo.core.expr.numeric_expr import (
    nonpyomo_leaf_types,
    NumericConstant,
    UnaryFunctionExpression,
    ProductExpression,
    ReciprocalExpression,
    PowExpression,
    SumExpression,
    LinearExpression,
    MonomialTermExpression,
    AbsExpression,
    NegationExpression,
)
from suspect.pyomo.expr_dict import ExpressionDict
from suspect.float_hash import BTreeFloatHasher
from galini.pyomo.postprocess import (
    detect_auxiliary_variables,
    detect_rlt_constraints,
)
from galini.pyomo.util import (
    model_variables,
    model_objectives,
    model_constraints,
)
import galini.core as core


def dag_from_pyomo_model(model):
    """Convert the Pyomo ``model`` to GALINI Problem.

    Parameters
    ----------
    model : ConcreteModel
        the Pyomo model.

    Returns
    -------
    galini.core.Problem
        GALINI problem.
    """
    warnings.warn(
        "dag_from_pyomo_model is deprecated, use problem_from_pyomo_model instead",
        DeprecationWarning,
    )
    return problem_from_pyomo_model(model)


def problem_from_pyomo_model(model):
    """Convert the Pyomo ``model`` to GALINI Problem.

    Parameters
    ----------
    model : ConcreteModel
        the Pyomo model.

    Returns
    -------
    galini.core.Problem
        GALINI problem.
    """
    if model.name:
        name = model.name
    else:
        name = 'unknown'
    problem = core.Problem(name)
    factory = _ComponentFactory(problem)
    for omo_var in model_variables(model):
        factory.add_variable(omo_var)

    for omo_cons in model_constraints(model):
        factory.add_constraint(omo_cons)

    for omo_obj in model_objectives(model):
        factory.add_objective(omo_obj)

    if not problem.objective:
        # Add constant objective
        problem.add_objective(
            core.Objective(
                '_constant_objective',
                core.Constant(0.0),
                core.Sense.MINIMIZE,
            ),
        )

    detect_auxiliary_variables(problem)
    detect_rlt_constraints(problem)

    return problem


class _ComponentFactory(object):
    def __init__(self, problem):
        self.problem = problem
        self._components = ExpressionDict(float_hasher=BTreeFloatHasher())
        self._visitor = _ConvertExpressionVisitor(self._components, self.problem)

    def add_variable(self, omo_var):
        """Convert and add variable to the problem."""
        comp = self._components.get(omo_var)
        if comp is not None:
            return comp
        domain = _convert_domain(omo_var.domain)

        new_var = self.problem.add_variable(
            core.Variable(omo_var.name, omo_var.lb, omo_var.ub, domain)
        )
        if omo_var.value:
            self.problem.set_starting_point(new_var, omo_var.value)
        self._components[omo_var] = new_var
        return new_var

    def add_constraint(self, omo_cons):
        """Convert and add constraint to the problem."""
        expr = omo_cons.body
        lower_bound = omo_cons.lower
        upper_bound = omo_cons.upper
        if lower_bound is not None:
            lower_bound = aml.value(lower_bound)
        if upper_bound is not None:
            upper_bound = aml.value(upper_bound)
        root_expr = self._expression(expr)
        constraint = self.problem.add_constraint(
            core.Constraint(
                omo_cons.name,
                root_expr,
                lower_bound,
                upper_bound,
            ),
        )
        return constraint

    def add_objective(self, omo_obj):
        """Convert and add objective to the problem."""

        if omo_obj.is_minimizing():
            sign = 1.0
            original_sense = core.Sense.MINIMIZE
        else:
            sign = -1.0
            original_sense = core.Sense.MAXIMIZE

        root_expr = self._expression(sign * omo_obj.expr)
        obj = self.problem.add_objective(
            core.Objective(
                omo_obj.name,
                root_expr,
                original_sense,
            ),
        )
        return obj

    def _expression(self, expr):
        return self._visitor.dfs_postorder_stack(expr)


_unary_func_name_to_expr_cls = {
    'sqrt': core.SqrtExpression,
    'exp': core.ExpExpression,
    'log': core.LogExpression,
    'sin': core.SinExpression,
    'cos': core.CosExpression,
    'tan': core.TanExpression,
    'asin': core.AsinExpression,
    'acos': core.AcosExpression,
    'atan': core.AtanExpression,
}


def _convert_as(expr_cls):
    return lambda _, v: expr_cls(v)


def _convert_unary_function(node, values):
    assert len(values) == 1
    expr_cls = _unary_func_name_to_expr_cls.get(node.getname(), None)
    if expr_cls is None:
        raise RuntimeError(
            'Unknown UnaryFunctionExpression type {}'.format(node.getname())
        )
    return expr_cls(values)


def _is_product_with_reciprocal(children):
    assert len(children) == 2
    a, b = children
    if isinstance(a, core.DivisionExpression):
        if isinstance(a.children[0], core.Constant):
            return a.children[0].value == 1.0
    if isinstance(b, core.DivisionExpression):
        if isinstance(b.children[0], core.Constant):
            return b.children[0].value == 1.0
    return False


def _is_product_constant_with_linear(children):
    assert len(children) == 2
    a, b = children
    if isinstance(a, core.Constant) and isinstance(b, core.LinearExpression):
        return True
    if isinstance(b, core.Constant) and isinstance(a, core.LinearExpression):
        return True
    return False


def _is_bilinear_product(children):
    if len(children) != 2:
        return False
    a, b = children
    if isinstance(a, core.Variable) and isinstance(b, core.Variable):
        return True
    if isinstance(a, core.Variable) and isinstance(b, core.LinearExpression):
        return len(b.children) == 1 and b.constant_term == 0.0
    if isinstance(a, core.LinearExpression) and isinstance(b, core.Variable):
        return len(a.children) == 1 and a.constant_term == 0.0
    return False


def _bilinear_variables_with_coefficient(children):
    assert len(children) == 2
    a, b = children
    if isinstance(a, core.Variable) and isinstance(b, core.Variable):
        return a, b, 1.0
    if isinstance(a, core.Variable) and isinstance(b, core.LinearExpression):
        assert len(b.children) == 1
        assert b.constant_term == 0.0
        vb = b.children[0]
        return a, vb, b.coefficient(vb)
    if isinstance(a, core.LinearExpression) and isinstance(b, core.Variable):
        assert len(a.children) == 1
        assert a.constant_term == 0.0
        va = a.children[0]
        return va, b, a.coefficient(va)


def _constant_with_linear(children):
    a, b = children
    if isinstance(a, core.Constant):
        assert isinstance(b, core.LinearExpression)
        constant = a.value
        linear = b
    else:
        assert isinstance(a, core.LinearExpression)
        assert isinstance(b, core.Constant)
        constant = b.value
        linear = a

    constant_term = linear.constant_term * constant
    variables = [v for v in linear.children]
    coefficients = [constant * linear.coefficient(v) for v in variables]
    return core.LinearExpression(variables, coefficients, constant_term)


def _reciprocal_product_numerator_denominator(children):
    a, b = children
    if isinstance(a, core.DivisionExpression):
        assert not isinstance(b, core.DivisionExpression)
        _, d = a.children
        return b, d

    assert isinstance(b, core.DivisionExpression)
    _, d = b.children
    return a, d


def _convert_product(_node, values):
    if _is_product_with_reciprocal(values):
        n, d = _reciprocal_product_numerator_denominator(values)
        return core.DivisionExpression([n, d])

    if _is_product_constant_with_linear(values):
        return _constant_with_linear(values)

    if _is_bilinear_product(values):
        a, b, c = _bilinear_variables_with_coefficient(values)
        return core.QuadraticExpression([a], [b], [c])

    return core.ProductExpression(values)


def _convert_reciprocal(_node, values):
    assert len(values) == 1
    return core.DivisionExpression([core.Constant(1.0), values[0]])


def _decompose_sum(children):
    quadratic = []
    linear = []
    constant = None
    other = []
    for child in children:
        if isinstance(child, core.QuadraticExpression):
            quadratic.append(child)
        elif isinstance(child, core.LinearExpression):
            linear.append(child)
        elif isinstance(child, core.Variable):
            linear.append(core.LinearExpression([child], [1.0], 0.0))
        elif isinstance(child, core.Constant):
            if constant is None:
                constant = 0.0
            constant += child.value
        else:
            other.append(child)

    if len(linear) > 0 and constant is not None:
        linear.append(core.LinearExpression([], [], constant))
    elif constant is not None:
        other.append(core.Constant(constant))

    return quadratic, linear, other


def _convert_sum(_node, values):
    quadratic, linear, other = _decompose_sum(values)

    if len(linear) == 0 and len(other) == 0:
        return core.QuadraticExpression(quadratic)

    if len(quadratic) == 0 and len(other) == 0:
        return core.LinearExpression(linear)

    children = other
    if len(quadratic) > 0:
        quadratic_expr = core.QuadraticExpression(quadratic)
        children.append(quadratic_expr)

    if len(linear) > 0:
        linear_expr = core.LinearExpression(linear)
        children.append(linear_expr)

    return core.SumExpression(children)


def _is_square(children):
    assert len(children) == 2
    base, expo = children
    if not isinstance(expo, core.Constant):
        return False

    if not (int(expo.value) == expo.value == 2.0):
        return False

    if isinstance(base, core.LinearExpression):
        return len(base.children) == 1 and base.constant_term == 0.0
    if isinstance(base, core.Variable):
        return True
    return False


def _square_variable_and_exponent(children):
    assert len(children) == 2
    base, expo = children
    if isinstance(base, core.LinearExpression):
        assert len(base.children) == 1
        return base.children[0], expo.value
    return base, expo.value


def _convert_pow(_node, values):
    if _is_square(values):
        var, expo = _square_variable_and_exponent(values)
        return core.QuadraticExpression([var], [var], [1.0])
    return core.PowExpression(values)


def _convert_monomial(_node, values):
    const, var = values
    assert isinstance(const, core.Constant)
    assert isinstance(var, core.Variable)
    return core.LinearExpression([var], [const.value], 0.0)


_convert_expr_map = dict()
_convert_expr_map[UnaryFunctionExpression] = _convert_unary_function
_convert_expr_map[ProductExpression] = _convert_product
_convert_expr_map[ReciprocalExpression] = _convert_reciprocal
_convert_expr_map[PowExpression] = _convert_pow
_convert_expr_map[SumExpression] = _convert_sum
_convert_expr_map[MonomialTermExpression] = _convert_monomial
_convert_expr_map[AbsExpression] = _convert_as(core.AbsExpression)
_convert_expr_map[NegationExpression] = _convert_as(core.NegationExpression)


class _ConvertExpressionVisitor(ExpressionValueVisitor):
    def __init__(self, memo, _problem):
        self.memo = memo

    def get(self, expr):
        """Get galini expression equivalent to expr."""
        if isinstance(expr, Number):
            const = NumericConstant(expr)
            return self.get(const)
        return self.memo[expr]

    def set(self, expr, new_expr):
        """Set expr to new_expr index."""
        self.memo.set(expr, new_expr)

    def visiting_potential_leaf(self, node):
        if node.__class__ in nonpyomo_leaf_types:
            expr = NumericConstant(float(node))
            const = core.Constant(float(node))
            self.set(expr, const)
            return True, const

        if node.is_variable_type():
            return True, self.get(node)

        # LinearExpression is special because it does not have
        # args even thought it has children.
        if isinstance(node, LinearExpression):
            variables = [self.get(var) for var in node.linear_vars]
            return True, core.LinearExpression(variables, node.linear_coefs, node.constant)

        polynomial_degree = node.polynomial_degree()
        if polynomial_degree == 1:
            linear_coefs = [0.0] * node.nargs()
            variables = [None] * node.nargs()
            for i, arg in enumerate(node.args):
                if arg.is_variable_type():
                    linear_coefs[i] = 1.0
                    variables[i] = self.get(arg)
                else:
                    coef, var = arg.args
                    linear_coefs[i] = coef
                    variables[i] = self.get(var)
            return True, core.LinearExpression(variables, linear_coefs, 0.0)

        if polynomial_degree == 2:
            linear = []
            quadratic = []
            constant = None
            for arg in node.args:
                if arg.__class__ in nonpyomo_leaf_types:
                    constant = arg
                    continue
                arg_polynomial_degree = arg.polynomial_degree()
                if arg_polynomial_degree == 1:
                    coef, var = arg.args
                    linear.append(
                        core.LinearExpression([self.get(var)], [coef], 0.0)
                    )
                elif arg_polynomial_degree == 2:
                    a, b = arg.args
                    xy_coef = 1.0
                    if a.is_variable_type():
                        x = self.get(a)
                    else:
                        coef, a = a.args
                        x = self.get(a)
                        xy_coef = coef
                    if b.is_variable_type():
                        y = self.get(b)
                    else:
                        coef, b = b.args
                        y = self.get(b)
                        xy_coef = coef
                    quadratic.append(
                        core.QuadraticExpression([x], [y], [xy_coef])
                    )
                else:
                    raise RuntimeError('Unexpected children {}'.format(arg))
            assert len(quadratic) > 0
            expr = core.QuadraticExpression(quadratic)
            if len(linear) > 0:
                children = [expr, core.LinearExpression(linear)]
                if constant is not None:
                    children.append(core.Constant(constant))
                expr = core.SumExpression(children)
            return True, expr
        return False, None

    def visit(self, node, values):
        if self.get(node) is not None:
            return self.get(node)

        callback = _convert_expr_map.get(type(node), None)
        if callback is None:
            raise RuntimeError('Unknown expression type {}'.format(type(node)))

        new_expr = callback(node, values)
        self.set(node, new_expr)
        return new_expr


def _convert_domain(dom):
    if isinstance(dom, aml.RealSet):
        return core.Domain.REAL
    elif isinstance(dom, aml.IntegerSet):
        return core.Domain.INTEGER
    elif isinstance(dom, aml.BooleanSet):
        return core.Domain.BINARY
    else:
        raise RuntimeError('Unknown domain {}'.format(dom))
