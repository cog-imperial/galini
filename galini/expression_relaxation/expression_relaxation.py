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

"""ExpressionRelaxation interface."""
from abc import ABCMeta, abstractmethod
from enum import IntEnum

from galini.core import Expression, Constraint


class RelaxationSide(IntEnum):
    """Represent the relaxation side."""
    UNDER = 0
    OVER = 1
    BOTH = 2


class ExpressionRelaxation(metaclass=ABCMeta):
    """Base class for expression_relaxation."""

    @abstractmethod
    def can_relax(self, problem, expr, ctx): # pragma: no cover
        """Check if the current relaxation can relax the expression.

        Parameters
        ----------
        problem : Problem
            the problem
        expr : Expression
            the expression
        ctx : ProblemContext
            a context object with information about special structure

        Returns
        -------
        bool
        """

    @abstractmethod
    def relax(self, problem, expr, ctx, **kwargs): # pragma: no cover
        """Return a relaxation for `expr`.

        Parameters
        ----------
        problem : Problem
            the problem
        expr : Expression
            the expression
        ctx : ProblemContext
            a context object with information about special structure

        Returns
        -------
        ExpressionRelaxationResult
        """


class ExpressionRelaxationResult:
    """Represent the result of an expression relaxation.

    The result contains a expression that relaxes the original expression
    and a set of additional constraints to be added to the problem.
    """
    def __init__(self, expression, constraints=None):
        if not isinstance(expression, Expression):
            raise ValueError('expression must be instance of Expression')

        if constraints is None:
            constraints = []

        for constraint in constraints:
            if not isinstance(constraint, Constraint):
                raise ValueError(
                    'constraints must contain values of type Constraint'
                )

        self.expression = expression
        self.constraints = constraints
