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
import numpy as np
import pyomo.environ as aml
from galini.pyomo.util import (
    model_variables,
    model_objectives,
    model_constraints,
    bounds_and_expr,
)
from galini.pyomo.expr_visitor import ExpressionHandler, bottom_up_visit
from galini.pyomo.expr_dict import ExpressionDict
from galini.pyomo.float_hash import BTreeFloatHasher
import galini.core as core


def dag_from_pyomo_model(model):
    """Convert the Pyomo ``model`` to GALINI DAG.

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
    dag = core.Problem(name=name)
    factory = _ComponentFactory(dag)
    for omo_var in model_variables(model):
        factory.add_variable(omo_var)

    for omo_cons in model_constraints(model):
        factory.add_constraint(omo_cons)

    for omo_obj in model_objectives(model):
        factory.add_objective(omo_obj)

    return dag


class _ComponentFactory(object):
    def __init__(self, dag):
        self._components = ExpressionDict(float_hasher=BTreeFloatHasher())
        self.dag = dag

    def add_variable(self, omo_var):
        """Convert and add variable to the problem."""
        comp = self._components.get(omo_var)
        if comp is not None:
            return comp
        domain = _convert_domain(omo_var.domain)
        new_var = self.dag.add_variable(omo_var.name, omo_var.lb, omo_var.ub, domain)
        if omo_var.value:
            new_var.set_starting_point(omo_var.value)
        self._components[omo_var] = new_var
        return new_var

    def add_constraint(self, omo_cons):
        """Convert and add constraint to the problem."""
        (lower_bound, upper_bound), expr = bounds_and_expr(omo_cons.expr)
        root_expr = self._expression(expr)
        constraint = self.dag.add_constraint(
            omo_cons.name,
            root_expr,
            lower_bound,
            upper_bound,
        )
        return constraint

    def add_objective(self, omo_obj):
        """Convert and add objective to the problem."""
        if omo_obj.is_minimizing():
            sense = core.Sense.MINIMIZE
        else:
            sense = core.Sense.MAXIMIZE

        root_expr = self._expression(omo_obj.expr)
        obj = self.dag.add_objective(
            omo_obj.name,
            root_expr,
            sense,
        )
        return obj

    def _expression(self, expr):
        return _convert_expression(self._components, self.dag, expr)


def _convert_domain(dom):
    if isinstance(dom, aml.RealSet):
        return core.Domain.REALS
    elif isinstance(dom, aml.IntegerSet):
        return core.Domain.INTEGERS
    elif isinstance(dom, aml.BooleanSet):
        return core.Domain.BINARY
    else:
        raise RuntimeError('Unknown domain {}'.format(dom))


def _convert_expression(memo, dag, expr):
    handler = _ExpressionConverterHandler(memo, dag)
    bottom_up_visit(handler, expr)
    return memo[expr]


# pylint: disable=protected-access
class _ExpressionConverterHandler(ExpressionHandler):
    def __init__(self, memo, dag):
        self.memo = memo
        self.dag = dag

    def get(self, expr):
        """Get galini expression equivalent to expr."""
        if isinstance(expr, Number):
            const = aml.NumericConstant(expr)
            return self.get(const)
        return self.memo[expr]

    def get_idx(self, expr):
        """Get core expression index."""
        core_expr = self.get(expr)
        if core_expr is not None:
            return core_expr.idx
        return None

    def set(self, expr, new_expr):
        """Set expr to new_expr index."""
        self.dag.insert_vertex(new_expr)
        self.memo.set(expr, new_expr)

    def _check_children(self, expr):
        for arg in expr._args:
            if self.get(arg) is None:
                raise RuntimeError('unknown child')

    def visit_number(self, n):
        const = aml.NumericConstant(n)
        self.visit_numeric_constant(const)

    def visit_numeric_constant(self, expr):
        if self.memo[expr] is not None:
            return self.memo[expr]
        const = core.Constant(expr.value)
        self.set(expr, const)
        return None

    def visit_variable(self, expr):
        if self.memo[expr] is not None:
            return self.memo[expr]
        raise AssertionError('Unknown variable encountered')

    def visit_equality(self, expr):
        raise AssertionError('Invalid EqualityExpression encountered')

    def visit_inequality(self, expr):
        raise AssertionError('Invalid EqualityExpression encountered')

    def visit_product(self, expr):
        if self.memo[expr] is not None:
            return self.memo[expr]
        self._check_children(expr)
        children = [self.get_idx(a) for a in expr._args]
        new_expr = core.ProductExpression(children)
        self.set(expr, new_expr)
        return None

    def visit_division(self, expr):
        if self.memo[expr] is not None:
            return self.memo[expr]

        self._check_children(expr)
        children = [self.get_idx(a) for a in expr._args]
        new_expr = core.DivisionExpression(children)
        self.set(expr, new_expr)
        return None

    def visit_sum(self, expr):
        if self.memo[expr] is not None:
            return self.memo[expr]

        self._check_children(expr)
        children = [self.get_idx(a) for a in expr._args]
        new_expr = core.SumExpression(children)
        self.set(expr, new_expr)
        return None

    def visit_linear(self, expr):
        if self.memo[expr] is not None:
            return self.memo[expr]
        self._check_children(expr)
        children = [self.get_idx(a) for a in expr._args]
        coeffs = np.array([expr._coef[id(a)] for a in expr._args], dtype=core.float_)
        const = expr._const
        new_expr = core.LinearExpression(
            children, coeffs, const
        )
        self.set(expr, new_expr)
        return None

    def visit_negation(self, expr):
        if self.memo[expr] is not None:
            return self.memo[expr]

        self._check_children(expr)
        children = [self.get_idx(a) for a in expr._args]
        new_expr = core.NegationExpression(children)
        self.set(expr, new_expr)
        return None

    def visit_unary_function(self, expr):
        if self.memo[expr] is not None:
            return self.memo[expr]

        self._check_children(expr)
        children = [self.get_idx(a) for a in expr._args]
        assert len(children) == 1
        fun = expr.name
        # pylint: disable=invalid-name
        ExprClass = {
            'sqrt': core.SqrtExpression,
            'exp': core.ExpExpression,
            'log': core.LogExpression,
            'sin': core.SinExpression,
            'cos': core.CosExpression,
            'tan': core.TanExpression,
            'asin': core.AsinExpression,
            'acos': core.AcosExpression,
            'atan': core.AtanExpression,
        }.get(fun)
        if ExprClass is None:
            raise AssertionError('Unknwon function', fun)
        new_expr = ExprClass(children)
        self.set(expr, new_expr)
        return None

    def visit_abs(self, expr):
        if self.memo[expr] is not None:
            return self.memo[expr]

        self._check_children(expr)
        children = [self.get_idx(a) for a in expr._args]
        new_expr = core.AbsExpression(children)
        self.set(expr, new_expr)
        return None

    def visit_pow(self, expr):
        if self.memo[expr] is not None:
            return self.memo[expr]

        self._check_children(expr)
        children = [self.get_idx(a) for a in expr._args]
        new_expr = core.PowExpression(children)
        self.set(expr, new_expr)
        return None
