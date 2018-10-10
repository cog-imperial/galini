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

"""Special structure detection module."""
from suspect.propagation import SpecialStructurePropagationVisitor
from suspect.polynomial import PolynomialDegreeVisitor
from suspect.fbbt import BoundsTightener, FBBTStopCriterion
from suspect.interval import Interval
from galini.suspect import ProblemContext, ExpressionDict, ProblemForwardIterator, ProblemBackwardIterator


def detect_polynomial_degree(problem, ctx=None):
    if ctx is None:
        ctx = ExpressionDict(problem)
    visitor = PolynomialDegreeVisitor()
    iterator = ProblemForwardIterator()
    iterator.iterate(problem, visitor, ctx)
    return ctx


def detect_special_structure(problem):
    ctx = ProblemContext(problem)
    bounds_tightener = BoundsTightener(
        ProblemForwardIterator(),
        ProblemBackwardIterator(),
        FBBTStopCriterion(),
    )

    # set bounds of root_expr to constraints bounds
    # since GALINI doesn't consider constraints as expression we have
    # to do this manually.
    for constraint in problem.constraints.values():
        expr_bounds = Interval(constraint.lower_bound, constraint.upper_bound)
        ctx.set_bounds(constraint.root_expr, expr_bounds)

    bounds_tightener.tighten(problem, ctx)
    detect_polynomial_degree(problem, ctx.polynomial)
    propagate_special_structure(problem, ctx)

    return ctx


def propagate_special_structure(problem, ctx=None):
    if ctx is None:
        ctx = ProblemContext(problem)
    visitor = SpecialStructurePropagationVisitor(problem)
    iterator = ProblemForwardIterator()
    iterator.iterate(problem, visitor, ctx)
    return ctx
