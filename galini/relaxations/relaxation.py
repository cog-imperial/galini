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
from galini.core import Expression, Constraint


class Relaxation(object):
    def can_relax(self, problem, expr, ctx):
        """Check if the current relaxation can relax the given expression.

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

    def relax(self, problem, expr, ctx):
        """Relax the given expression.

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
        RelaxationResult
        """
        pass


class RelaxationResult(object):
    """Represent the result of a relaxation.

    The result contains a relaxed expression and a set of additional
    constraints to be added to the problem.
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
