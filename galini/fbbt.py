# Copyright 2019 Francesco Ceccon
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

"""Perform FBBT on a Galini problem."""
from suspect.fbbt.initialization import BoundsInitializationVisitor
from suspect.fbbt.propagation import BoundsPropagationVisitor
import suspect.fbbt.propagation.rules as prop
from suspect.fbbt.tightening import BoundsTighteningVisitor
import suspect.fbbt.tightening.rules as tight
from galini.suspect import ProblemForwardIterator, ProblemBackwardIterator
import galini.core as core


_expr_to_prop = dict()
_expr_to_prop[core.Variable] = prop.VariableRule()
_expr_to_prop[core.AuxiliaryVariable] = prop.VariableRule()
_expr_to_prop[core.Constant] = prop.ConstantRule()
_expr_to_prop[core.LinearExpression] = prop.LinearRule()
_expr_to_prop[core.QuadraticExpression] = prop.QuadraticRule()
_expr_to_prop[core.SumExpression] = prop.SumRule()
_expr_to_prop[core.NegationExpression] = prop.NegationRule()
_expr_to_prop[core.ProductExpression] = prop.ProductRule()

class _GaliniBoundsPropagationVisitor(BoundsPropagationVisitor):
    def visit_expression(self, expr, bounds):
        rule = _expr_to_prop[type(expr)]
        return True, rule.apply(expr, bounds)


_expr_to_tight = dict()
_expr_to_tight[core.LinearExpression] = tight.LinearRule()
_expr_to_tight[core.QuadraticExpression] = tight.QuadraticRule()
_expr_to_tight[core.SumExpression] = tight.SumRule()

class _GaliniBoundsTighteningVisitor(BoundsTighteningVisitor):
    def visit_expression(self, expr, bounds):
        rule = _expr_to_tight.get(type(expr))
        if rule is not None:
            result = rule.apply(expr, bounds)
        else:
            result = None

        if result is not None:
            return True, result
        return False, None


class BoundsTightener(object):
    """Configure and run FBBT on a problem.

    Parameters
    ----------
    forward_iterator:
       forward iterator over vertices of the problem
    backward_iterator:
       backward iterator over vertices of the problem
    stop_criterion:
       criterion used to stop iteration
    """
    def __init__(self, stop_criterion):
        self._forward_iterator = ProblemForwardIterator()
        self._backward_iterator = ProblemBackwardIterator()
        self._stop_criterion = stop_criterion

    def tighten(self, problem, bounds):
        """Tighten bounds of ``problem`` storing them in ``bounds``."""
        self._forward_iterator.iterate(problem, BoundsInitializationVisitor(), bounds)
        if self._stop_criterion._max_iter == 0:
            return

        prop_visitor = self._stop_criterion.intercept_changes(_GaliniBoundsPropagationVisitor())
        tigh_visitor = self._stop_criterion.intercept_changes(_GaliniBoundsTighteningVisitor())
        changes_tigh = None
        changes_prop = None
        while not self._stop_criterion.should_stop():
            changes_prop = self._forward_iterator.iterate(
                problem, prop_visitor, bounds, starting_vertices=changes_tigh
            )
            changes_tigh = self._backward_iterator.iterate(
                problem, tigh_visitor, bounds, starting_vertices=changes_prop
            )
            if len(changes_prop) == 0 and len(changes_tigh) == 0:
                return
            self._stop_criterion.iteration_end()
