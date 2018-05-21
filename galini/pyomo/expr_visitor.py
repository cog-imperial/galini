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
"""Visitor for Pyomo expression trees."""
from typing import Any, Dict, List, Tuple
from abc import ABC, abstractmethod
from numbers import Number
from pyomo.core.base import Var
from pyomo.core.base.var import (_GeneralVarData, SimpleVar)
import pyomo.core.base.expr_pyomo4 as omo


Variable = Var # pylint: disable=invalid-name
NumericConstant = omo.NumericConstant # pylint: disable=invalid-name
Expression = omo._ExpressionBase # pylint: disable=invalid-name,protected-access
EqualityExpression = omo._EqualityExpression # pylint: disable=invalid-name,protected-access
InequalityExpression = omo._InequalityExpression # pylint: disable=invalid-name,protected-access
ProductExpression = omo._ProductExpression # pylint: disable=invalid-name,protected-access
DivisionExpression = omo._DivisionExpression # pylint: disable=invalid-name,protected-access
SumExpression = omo._SumExpression # pylint: disable=invalid-name,protected-access
LinearExpression = omo._LinearExpression # pylint: disable=invalid-name,protected-access
NegationExpression = omo._NegationExpression # pylint: disable=invalid-name,protected-access
UnaryFunctionExpression = omo._UnaryFunctionExpression # pylint: disable=invalid-name,protected-access
AbsExpression = omo._AbsExpression # pylint: disable=invalid-name,protected-access
PowExpression = omo._PowExpression # pylint: disable=invalid-name,protected-access


CALLBACKS_NAMES: Dict[Any, str] = {
    _GeneralVarData: 'visit_variable',
    SimpleVar: 'visit_variable',
    Variable: 'visit_variable',
    int: 'visit_number',
    float: 'visit_number',
    Number: 'visit_number',
    NumericConstant: 'visit_numeric_constant',
    EqualityExpression: 'visit_equality',
    InequalityExpression: 'visit_inequality',
    ProductExpression: 'visit_product',
    DivisionExpression: 'visit_division',
    SumExpression: 'visit_sum',
    LinearExpression: 'visit_linear',
    NegationExpression: 'visit_negation',
    UnaryFunctionExpression: 'visit_unary_function',
    AbsExpression: 'visit_abs',
    PowExpression: 'visit_pow',

}


class ExpressionHandler(ABC):
    """Visitor for Pyomo expression trees."""
    @abstractmethod
    def visit_number(self, n: Number) -> None:
        """Visit number."""
        pass

    @abstractmethod
    def visit_numeric_constant(self, expr: Any) -> None:
        """Visit numeric constant."""
        pass

    @abstractmethod
    def visit_variable(self, expr: Any) -> None:
        """Vist variable."""
        pass

    @abstractmethod
    def visit_equality(self, expr: Any) -> None:
        """Visit equality expression."""
        pass

    @abstractmethod
    def visit_inequality(self, expr: Any) -> None:
        """Visit inequality expression."""
        pass

    @abstractmethod
    def visit_product(self, expr: Any) -> None:
        """Visit product expression."""
        pass

    @abstractmethod
    def visit_division(self, expr: Any) -> None:
        """Visit division expression."""
        pass

    @abstractmethod
    def visit_sum(self, expr: Any) -> None:
        """Visit sum expression."""
        pass

    @abstractmethod
    def visit_linear(self, expr: Any) -> None:
        """Visit linear expression."""
        pass

    @abstractmethod
    def visit_negation(self, expr: Any) -> None:
        """Visit negation expression."""
        pass

    @abstractmethod
    def visit_unary_function(self, expr: Any) -> None:
        """Visit unary function expression."""
        pass

    @abstractmethod
    def visit_abs(self, expr: Any) -> None:
        """Visit abs expression."""
        pass

    @abstractmethod
    def visit_pow(self, expr: Any) -> None:
        """Vist pow expression."""
        pass


def try_callback(handler: ExpressionHandler, expr: Any) -> None:
    """Try calling one of the registered callbacks, raising an exception
    if no callback matches.
    """
    expr_type = type(expr)
    callback_name = CALLBACKS_NAMES.get(expr_type)
    if callback_name is None:
        msg = 'No callback found for {} {}'.format(
            expr, type(expr)
        )
        raise RuntimeError(msg)
    else:
        callback = getattr(handler, callback_name)
        if callback is None:
            raise RuntimeError('Missing callback {}'.format(callback_name))
        callback(expr)


def bottom_up_visit(handler: ExpressionHandler, expr: Any) -> None:
    """Visit the expression from leaf nodes to root.

    Parameters
    ----------
    handler: ExpressionHandler
        handler that will handle each sub-expression callback
    expr:
        the expression to visit
    """
    expr_level: Dict[int, int] = {}
    expr_by_id: Dict[int, Any] = {}
    with omo.bypass_clone_check():
        stack = [(0, expr)]
        while stack:
            (lvl, expr) = stack.pop()

            old_lvl = expr_level.get(id(expr), -1)
            expr_level[id(expr)] = max(old_lvl, lvl)
            expr_by_id[id(expr)] = expr

            if isinstance(expr, omo._ExpressionBase):
                for arg in expr._args:
                    stack.append((lvl+1, arg))

        expr_level_tuple: List[Tuple[int, int]] = sorted(
            [(lvl, ex) for ex, lvl in expr_level.items()],
            reverse=True,
        )

        for _, expr_id in expr_level_tuple:
            expr = expr_by_id[expr_id]
            try_callback(handler, expr)
