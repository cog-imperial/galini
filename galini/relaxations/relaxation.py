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

"""Relaxation interface."""
from abc import ABCMeta, abstractmethod
from suspect.expression import ExpressionType, UnaryFunctionType
from galini.core import (
    RelaxedProblem,
    Constraint,
    Objective,
)
import galini.core as core


class Relaxation(metaclass=ABCMeta):
    """Relaxation of a problem.

    Given an optimization problem, returns a new relaxed problem.
    """
    def __init__(self):
        self._problem_expr = {}

    def relax(self, problem):
        self.before_relax(problem)

        relaxed_problem = RelaxedProblem(
            self.relaxed_problem_name(problem),
            problem,
        )

        for obj_name, obj in problem.objectives.items():
            self._relax_objective(problem, relaxed_problem, obj_name, obj)

        for cons_name, cons in problem.constraints.items():
            self._relax_constraint(problem, relaxed_problem, cons_name, cons)

        self.after_relax(problem, relaxed_problem)
        return relaxed_problem

    @abstractmethod
    def relaxed_problem_name(self, problem): # pragma: no cover
        """Return the name of the `problem` relaxation.

        Parameters
        ----------
        problem : Problem

        Returns
        -------
        str
        """
        pass

    @abstractmethod
    def relax_objective(self, objective): # pragma: no cover
        """Relax the `objective`.

        Parameters
        ----------
        objective : Objective

        Returns
        -------
        ObjectiveRelaxationResult
        """
        pass

    @abstractmethod
    def relax_constraint(self, constraint): # pragma: no cover
        """Relax the `constraint`.

        Parameters
        ----------
        constraint : Constraint

        Returns
        -------
        ConstraintRelaxationResult
        """
        pass

    def before_relax(self, problem):
        """Callback executed before relaxing the problem."""
        pass

    def after_relax(self, problem, relaxed_problem):
        """Callback executed after relaxing the problem."""

    def _relax_objective(self, problem, relaxed_problem, obj_name, obj):
        result = self.relax_objective(obj)
        if not isinstance(result, ObjectiveRelaxationResult):
            raise ValueError('relax_objective must return object of type ObjectiveRelaxationResult')
        new_obj = result.objective
        new_expr = self._insert_expression(new_obj.root_expr, problem, relaxed_problem)
        return relaxed_problem.add_objective(
            new_obj.name,
            new_expr,
            new_obj.sense,
        )

    def _relax_constraint(self, problem, relaxed_problem, cons_name, cons):
        result = self.relax_constraint(cons)
        if not isinstance(result, ConstraintRelaxationResult):
            raise ValueError('relax_constraint must return object of type ConstraintRelaxationResult')
        new_cons = result.constraint
        new_expr = self._insert_expression(new_cons.root_expr, problem, relaxed_problem)
        return relaxed_problem.add_constraint(
            new_cons.name,
            new_expr,
            new_cons.lower_bound,
            new_cons.upper_bound,
        )

    def _insert_expression(self, expr, problem, relaxed_problem):
        def _inner(expr):
            if expr.problem is not None and expr.idx in self._problem_expr:
                return self._problem_expr[expr.idx]

            if expr.expression_type == ExpressionType.Variable:
                new_var = relaxed_problem.add_variable(
                    expr.name,
                    expr.lower_bound,
                    expr.upper_bound,
                    None,
                )
                self._problem_expr[expr.idx] = new_var
                return new_var
            else:
                children = [_inner(child) for child in expr.children]
                new_expr = _clone_expression(expr, children)
                self._problem_expr[expr.idx] = new_expr
                return new_expr
        new_expr = _inner(expr)
        relaxed_problem.insert_tree(new_expr)
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


def _clone_expression(expr, children):
    type_ = expr.expression_type
    if type_ == ExpressionType.Linear:
        return core.LinearExpression(children, expr.coefficients, expr.constant_term)
    elif type_ == ExpressionType.Constant:
        return core.Constant(expr.value)
    elif type_ == ExpressionType.UnaryFunction:
        func_type = expr.func_type
        cls = _FUNC_TYPE_TO_CLS[func_type]
        return cls(children)
    else:
        cls = _EXPR_TYPE_TO_CLS[type_]
        return cls(children)


class RelaxationResult(metaclass=ABCMeta):
    """Represents the result of a relaxation."""
    def __new__(cls, expression, *args, **kwargs):
        if isinstance(expression, Constraint):
            return super().__new__(ConstraintRelaxationResult)
        if isinstance(expression, Objective):
            return super().__new__(ObjectiveRelaxationResult)
        return super().__new__(cls)

    def __init__(self, expression, constraints=None):
        if not isinstance(expression, self.expr_cls):
            raise ValueError('expression must be instance of {}'.format(self.expr_cls))

        if constraints is None:
            constraints = []

        for constraint in constraints:
            if not isinstance(constraint, Constraint):
                raise ValueError('constraints must contain values of type Constraint')

        self.expression = expression
        self.constraints = constraints


class ConstraintRelaxationResult(RelaxationResult):
    """Represents the result of a constraint relaxation."""
    expr_cls = Constraint

    @property
    def constraint(self):
        return self.expression


class ObjectiveRelaxationResult(RelaxationResult):
    """Represents the result of an objective relaxation."""
    expr_cls = Objective

    @property
    def objective(self):
        return self.expression
