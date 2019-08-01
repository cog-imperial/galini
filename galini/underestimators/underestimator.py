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

"""Underestimator interface."""
from abc import ABCMeta, abstractmethod
from galini.core import Expression, Constraint


class Underestimator(metaclass=ABCMeta):
    @abstractmethod
    def can_underestimate(self, problem, expr, ctx): # pragma: no cover
        """Check if the current underestimator can underestimate the given expression.

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
        pass

    @abstractmethod
    def underestimate(self, problem, expr, ctx, **kwargs): # pragma: no cover
        """Return expression underestimating expr.

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
        UnderestimatorResult
        """
        pass


class UnderestimatorResult(object):
    """Represent the result of an underestimator.

    The result contains a expression that underestimates the original expression
    and a set of additional constraints to be added to the problem.
    """
    def __init__(self, expression, constraints=None):
        if not isinstance(expression, Expression):
            raise ValueError('expression must be instance of Expression')

        if constraints is None:
            constraints = []

        for constraint in constraints:
            if not isinstance(constraint, Constraint):
                raise ValueError('constraints must contain values of type Constraint')

        self.expression = expression
        self.constraints = constraints
