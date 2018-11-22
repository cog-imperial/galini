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

"""Underestimator for sum of other expressions."""
from suspect.expression import ExpressionType
from galini.core import SumExpression
from galini.underestimators.underestimator import Underestimator, UnderestimatorResult



class SumOfUnderestimators(Underestimator):
    def __init__(self, underestimators):
        for underestimator in underestimators:
            if not isinstance(underestimator, Underestimator):
                raise ValueError('All underestimators must be instances of Underestimator')
        self._underestimators = underestimators

    def can_underestimate(self, problem, expr, ctx):
        return (
            self._can_be_underestimated_as_sum_of_expressions(problem, expr, ctx) or
            self._can_be_underestimated_by_child_underestimator(problem, expr, ctx)
        )

    def underestimate(self, problem, expr, ctx):
        if self._can_be_underestimated_as_sum_of_expressions(problem, expr, ctx):
            return self._underestimate_as_sum(problem, expr, ctx)
        for underestimator in self._underestimators:
            if underestimator.can_underestimate(problem, expr, ctx):
                return underestimator.underestimate(problem, expr, ctx)
        return None

    def _can_be_underestimated_as_sum_of_expressions(self, problem, expr, ctx):
        if expr.expression_type == ExpressionType.Sum:
            for child in expr.children:
                for underestimator in self._underestimators:
                    if not underestimator.can_underestimate(problem, expr, ctx):
                        return False
            return True
        return False

    def _can_be_underestimated_by_child_underestimator(self, problem, expr, ctx):
        for underestimator in self._underestimators:
            if underestimator.can_underestimate(problem, expr, ctx):
                return True
        return False

    def _underestimate_as_sum(self, problem, expr, ctx):
        new_children = []
        new_constraints = []
        for child in expr.children:
            for underestimator in self._underestimators:
                if underestimator.can_underestimate(problem, expr, ctx):
                    result = underestimator.underestimate(problem, expr, ctx)
                    new_children.append(result.expression)
                    new_constraints.extend(result.constraints)
                    break
        new_expression = SumExpression(new_children)
        return UnderestimatorResult(new_expression, new_constraints)
