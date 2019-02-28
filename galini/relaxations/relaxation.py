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
    Variable,
    AuxiliaryVariable,
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

    def relax(self, problem, **kwargs):
        self._problem_expr = {}
        self.before_relax(problem, **kwargs)

        relaxed_problem = problem.make_relaxed(self.relaxed_problem_name(problem));

        for var in problem.variables:
            self._relax_variable(problem, relaxed_problem, var)

        for obj in problem.objectives:
            self._relax_objective(problem, relaxed_problem, obj)

        for cons in problem.constraints:
            self._relax_constraint(problem, relaxed_problem, cons)

        self.after_relax(problem, relaxed_problem, **kwargs)
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

    def relax_variable(self, problem, variable): # pragma: no cover
        """Relax the `variable`.

        Parameters
        ----------
        problem : Problem
        variable : Variable

        Returns
        -------
        Variable
        """
        pass

    @abstractmethod
    def relax_objective(self, problem, objective): # pragma: no cover
        """Relax the `objective`.

        Parameters
        ----------
        problem : Problem
        objective : Objective

        Returns
        -------
        ObjectiveRelaxationResult
        """
        pass

    @abstractmethod
    def relax_constraint(self, problem, constraint): # pragma: no cover
        """Relax the `constraint`.

        Parameters
        ----------
        problem : Problem
        constraint : Constraint

        Returns
        -------
        ConstraintRelaxationResult
        """
        pass

    def before_relax(self, problem, **kwargs):
        """Callback executed before relaxing the problem."""
        pass

    def after_relax(self, problem, relaxed_problem, **kwargs):
        """Callback executed after relaxing the problem."""
        pass

    def _relax_variable(self, problem, relaxed_problem, var):
        result = self.relax_variable(problem, var)
        if result is not None and not isinstance(result, Variable):
            raise ValueError('relax_variable must return object of type Variable or None')

        var_relaxed = relaxed_problem.variable(var.idx)
        self._problem_expr[var.uid] = var_relaxed

        if result is not None:
            # overwrite bounds and domain
            relaxed_problem.set_lower_bound(var_relaxed, result.lower_bound)
            relaxed_problem.set_upper_bound(var_relaxed, result.upper_bound)
            relaxed_problem.set_domain(var_relaxed, result.domain)

    def _relax_objective(self, problem, relaxed_problem, obj):
        result = self.relax_objective(problem, obj)
        if not isinstance(result, ObjectiveRelaxationResult):
            raise ValueError('relax_objective must return object of type ObjectiveRelaxationResult')
        new_obj = result.objective
        new_expr = self._insert_expression(new_obj.root_expr, problem, relaxed_problem)
        relaxed_problem.add_objective(
            new_obj.name,
            new_expr,
            new_obj.sense,
        )
        self._insert_constraints(result.constraints, problem, relaxed_problem)

    def _relax_constraint(self, problem, relaxed_problem, cons):
        result = self.relax_constraint(problem, cons)
        if not isinstance(result, ConstraintRelaxationResult):
            raise ValueError('relax_constraint must return object of type ConstraintRelaxationResult')
        new_cons = result.constraint
        new_expr = self._insert_expression(new_cons.root_expr, problem, relaxed_problem)
        relaxed_problem.add_constraint(
            new_cons.name,
            new_expr,
            new_cons.lower_bound,
            new_cons.upper_bound,
        )
        self._insert_constraints(result.constraints, problem, relaxed_problem)

    def _insert_variable(self, expr, problem, relaxed_problem, use_problem_bounds=False):
        assert expr.expression_type == ExpressionType.Variable
        if use_problem_bounds and expr.problem is not None:
            lower_bound = problem.lower_bound(expr)
            upper_bound = problem.upper_bound(expr)
            domain = problem.domain(expr)
        else:
            lower_bound = expr.lower_bound
            upper_bound = expr.upper_bound
            domain = expr.domain
        if isinstance(expr, AuxiliaryVariable):
            new_var = relaxed_problem.add_aux_variable(
                expr.name, lower_bound, upper_bound, domain, expr.reference)
        else:
            new_var = relaxed_problem.add_variable(expr.name, lower_bound, upper_bound, domain)
        self._problem_expr[expr.uid] = new_var
        return new_var

    def _insert_expression(self, expr, problem, relaxed_problem):
        def _inner(expr):
            if expr.uid in self._problem_expr:
                return self._problem_expr[expr.uid]

            if expr.expression_type == ExpressionType.Variable:
                return self._insert_variable(expr, problem, relaxed_problem, use_problem_bounds=True)
            else:
                children = [_inner(child) for child in expr.children]
                new_expr = _clone_expression(expr, children)
                self._problem_expr[expr.uid] = new_expr
                return new_expr
        new_expr = _inner(expr)
        relaxed_problem.insert_tree(new_expr)
        return new_expr

    def _insert_constraints(self, new_constraints, problem, relaxed_problem):
        for constraint in new_constraints:
            new_expr = self._insert_expression(constraint.root_expr, problem, relaxed_problem)
            x = relaxed_problem.add_constraint(
                constraint.name,
                new_expr,
                constraint.lower_bound,
                constraint.upper_bound,
            )


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
