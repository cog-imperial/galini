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
import pkg_resources
import suspect.convexity.rules as cvx_rules
import suspect.monotonicity.rules as mono_rules
from suspect.convexity import ConvexityPropagationVisitor
from suspect.interval import Interval
from suspect.monotonicity import MonotonicityPropagationVisitor
from suspect.propagation import SpecialStructurePropagationVisitor

import galini.core as core
from galini.fbbt import (
    BoundsTightener,
    FBBTStopCriterion,
    _GaliniBoundsPropagationVisitor,
)
from galini.suspect import (
    ProblemContext,
    ExpressionDict,
    ProblemForwardIterator,
)
from galini.timelimit import timeout


def perform_fbbt(problem, maxiter, timelimit, objective_upper_bound=None):
    """Perform FBBT on `problem` with the given `maxiter` and `timelimit`."""
    bounds = ExpressionDict(problem)
    bounds_tightener = BoundsTightener(
        FBBTStopCriterion(max_iter=maxiter, timelimit=timelimit),
    )

    for variable in problem.variables:
        lb = problem.lower_bound(variable)
        ub = problem.upper_bound(variable)
        bounds[variable] = Interval(lb, ub)

    if objective_upper_bound is not None:
        root_expr = problem.objective.root_expr
        expr_bounds = Interval(None, objective_upper_bound)
        if root_expr not in bounds:
            bounds[root_expr] = expr_bounds
        else:
            existing_bounds = bounds[root_expr]
            new_bounds = existing_bounds.intersect(expr_bounds)
            bounds[root_expr] = new_bounds

    # set bounds of root_expr to constraints bounds
    # since GALINI doesn't consider constraints as expression we have
    # to do this manually.
    for constraint in problem.constraints:
        root_expr = constraint.root_expr
        expr_bounds = Interval(constraint.lower_bound, constraint.upper_bound)
        if root_expr not in bounds:
            bounds[root_expr] = expr_bounds
        else:
            existing_bounds = bounds[root_expr]
            new_bounds = existing_bounds.intersect(expr_bounds)
            bounds[root_expr] = new_bounds

    try:
        with timeout(timelimit, 'Timeout in FBBT'):
            bounds_tightener.tighten(problem, bounds)
    except TimeoutError:
        pass

    return bounds


def detect_special_structure(problem, maxiter, timelimit):
    """Perform special structure detection on `problem`."""
    ctx = ProblemContext(problem)
    bounds_tightener = BoundsTightener(
        FBBTStopCriterion(max_iter=maxiter, timelimit=timelimit),
    )

    # set bounds of root_expr to constraints bounds
    # since GALINI doesn't consider constraints as expression we have
    # to do this manually.
    for constraint in problem.constraints:
        root_expr = constraint.root_expr
        expr_bounds = Interval(constraint.lower_bound, constraint.upper_bound)
        if root_expr not in ctx.bounds:
            ctx.set_bounds(root_expr, expr_bounds)
        else:
            existing_bounds = ctx.get_bounds(root_expr)
            new_bounds = existing_bounds.intersect(expr_bounds)
            ctx.set_bounds(root_expr, new_bounds)

    bounds_tightener.tighten(problem, ctx.bounds)
    propagate_special_structure(problem, ctx)

    return ctx


def propagate_special_structure(problem, bounds=None):
    """Propagate special structure, without perform FBBT."""
    visitor = _GaliniSpecialStructurePropagationVisitor(problem)
    iterator = ProblemForwardIterator()
    convexity = ExpressionDict(problem)
    monotonicity = ExpressionDict(problem)

    if bounds is None:
        bounds = ExpressionDict(problem)
        prop_visitor = _GaliniBoundsPropagationVisitor()
        iterator.iterate(problem, prop_visitor, bounds)

    iterator.iterate(problem, visitor, convexity, monotonicity, bounds)
    return bounds, monotonicity, convexity


_expr_to_mono = dict() # pylint: disable=invalid-name
_expr_to_mono[core.Variable] = mono_rules.VariableRule()
_expr_to_mono[core.Constant] = mono_rules.ConstantRule()
_expr_to_mono[core.LinearExpression] = mono_rules.LinearRule()
_expr_to_mono[core.QuadraticExpression] = mono_rules.QuadraticRule()
_expr_to_mono[core.SumExpression] = mono_rules.SumRule()
_expr_to_mono[core.NegationExpression] = mono_rules.NegationRule()


class _GaliniMonotonicityPropagationVisitor(MonotonicityPropagationVisitor):
    def visit_expression(self, expr, mono, bounds):
        rule = _expr_to_mono[type(expr)]
        return True, rule.apply(expr, mono, bounds)


_expr_to_cvx = dict() # pylint: disable=invalid-name
_expr_to_cvx[core.Variable] = cvx_rules.VariableRule()
_expr_to_cvx[core.Constant] = cvx_rules.ConstantRule()
_expr_to_cvx[core.LinearExpression] = cvx_rules.LinearRule()
_expr_to_cvx[core.QuadraticExpression] = cvx_rules.QuadraticRule()
_expr_to_cvx[core.SumExpression] = cvx_rules.SumRule()
_expr_to_cvx[core.NegationExpression] = cvx_rules.NegationRule()


class _GaliniConvexityPropagationVisitor(ConvexityPropagationVisitor):
    # pylint: disable=arguments-differ
    def visit_expression(self, expr, cvx, mono, bounds):
        rule = _expr_to_cvx[type(expr)]
        return True, rule.apply(expr, cvx, mono, bounds)


def _monotonicity_detection_entry_points():
    return pkg_resources.iter_entry_points('suspect.monotonicity_detection')


def _convexity_detection_entry_points():
    return pkg_resources.iter_entry_points('suspect.convexity_detection')


class _GaliniSpecialStructurePropagationVisitor(SpecialStructurePropagationVisitor): # pylint: disable=line-too-long
    def __init__(self, problem):
        super().__init__(problem)
        self._mono_visitors = [_GaliniMonotonicityPropagationVisitor()]
        for entry_point in _monotonicity_detection_entry_points():
            cls = entry_point.load()
            self._mono_visitors.append(cls(problem))

        self._cvx_visitors = [_GaliniConvexityPropagationVisitor()]
        for entry_point in _convexity_detection_entry_points():
            cls = entry_point.load()
            self._cvx_visitors.append(cls(problem))
