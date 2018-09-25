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


class Relaxation(object):
    def can_relax(self, expr, ctx):
        """Check if the current relaxation can relax the given expression.

        Parameters
        ----------
        expr : Expression
            the expression
        ctx : ProblemContext
            a context object with information about special structure

        Returns
        -------
        bool
        """
        pass

    def relax(self, expr, ctx):
        """Relax the given expression.

        Parameters
        ----------
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
    def __init__(self, expression, constraints):
        self.expression = expression
        self.constraints = constraints


class NewVariable(object):
    """Represent a new variable that will be added to the problem.

    Relaxations can reference existing variable but, if they neeed to do so,
    can introduce new variables in the problem by instantiating an object of
    this class.
    """
    pass
