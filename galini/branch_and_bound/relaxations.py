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

"""Contain relaxations used in the Branch & Cut algorithm."""
from galini.core import (
    Domain,
    Objective,
    Constraint,
    Variable,
    LinearExpression,
    SumExpression,
)
from galini.relaxations import Relaxation, RelaxationResult
from galini.suspect import ProblemContext
from galini.expression_relaxation import (
    McCormickExpressionRelaxation,
    LinearExpressionRelaxation,
    DisaggregateBilinearExpressionRelaxation,
    SumOfUnderestimators,
    RelaxationSide,
)
from galini.expression_relaxation.bilinear import (
    BILINEAR_AUX_VAR_META,
)


class _RelaxationBase(Relaxation):
    def __init__(self, problem, bounds, monotonicity, convexity):
        super().__init__()
        self._ctx = ProblemContext(problem, bounds, monotonicity, convexity)
        self._underestimator = self._root_underestimator()

    def _root_underestimator(self):
        raise NotImplementedError('_root_underestimator')

    def before_relax(self, problem, relaxed_problem, **kwargs):
        # Copy bilinear aux var metadata to avoid duplicate aux vars
        if BILINEAR_AUX_VAR_META in problem.metadata:
            original_bilinear_aux = problem.metadata[BILINEAR_AUX_VAR_META]
            relaxed_bilinear_aux = dict()

            for xy_tuple, var in original_bilinear_aux.items():
                relaxed_var = relaxed_problem.variable(var)
                relaxed_bilinear_aux[xy_tuple] = relaxed_var

            relaxed_problem.metadata[BILINEAR_AUX_VAR_META] = \
                relaxed_bilinear_aux

        self._ctx.metadata = relaxed_problem.metadata
        self._before_relax(problem)

    def _before_relax(self, problem):
        pass

    def after_relax(self, problem, relaxed_problem, **kwargs):
        pass

    def relax_objective(self, problem, objective):
        result = self.relax_expression(
            problem, objective.root_expr, side=RelaxationSide.UNDER
        )
        new_objective = Objective(
            objective.name, result.expression, objective.original_sense
        )
        return RelaxationResult(new_objective, result.constraints)

    def relax_constraint(self, problem, constraint):
        if constraint.lower_bound is None:
            side = RelaxationSide.UNDER
        elif constraint.upper_bound is None:
            side = RelaxationSide.OVER
        else:
            side = RelaxationSide.BOTH

        result = self.relax_expression(
            problem, constraint.root_expr, side=side
        )
        new_constraint = Constraint(
            constraint.name,
            result.expression,
            constraint.lower_bound,
            constraint.upper_bound
        )
        new_constraint.metadata = constraint.metadata
        return RelaxationResult(new_constraint, result.constraints)

    def relax_expression(self, problem, expr, side=None):
        """Return relaxation of `expr`."""
        return self._relax_expression(problem, expr, side=side)

    def _relax_expression(self, problem, expr, side=None):
        assert self._underestimator.can_relax(problem, expr, self._ctx)
        result = self._underestimator.relax(
            problem, expr, self._ctx, side=side
        )
        return result


class ConvexRelaxation(_RelaxationBase):
    """Create a Convex relaxation of a (quadratic) nonconvex problem."""
    def _root_underestimator(self):
        return SumOfUnderestimators([
            LinearExpressionRelaxation(),
            DisaggregateBilinearExpressionRelaxation(),
        ])

    def relaxed_problem_name(self, problem):
        return problem.name + '_convex'

    def relax_expression(self, problem, expr, side=None):
        return self._relax_expression(problem, expr, side=side)


class LinearRelaxation(_RelaxationBase):
    """Create a linear relaxation of a convex problem."""
    def __init__(self, problem, bounds, monotonicity, convexity):
        super().__init__(problem, bounds, monotonicity, convexity)
        self._objective_count = 0

    def _before_relax(self, problem):
        self._objective_count = 0

    def _root_underestimator(self):
        return SumOfUnderestimators([
            LinearExpressionRelaxation(),
            McCormickExpressionRelaxation(linear=True),
        ])

    def relaxed_problem_name(self, problem):
        return problem.name + '_linear'

    def relax_objective(self, problem, objective):
        self._objective_count += 1
        if self._objective_count > 1:
            raise ValueError('Apply LinearRelaxation to multiobjective problem')
        new_variable = Variable('_objvar', None, None, Domain.REAL)
        new_objective_expr = LinearExpression([new_variable], [1.0], 0.0)
        new_objective = Objective(
            objective.name,
            new_objective_expr,
            objective.original_sense,
        )

        under_result = self.relax_expression(problem, objective.root_expr)

        new_cons_expr = SumExpression([
            under_result.expression,
            LinearExpression([new_variable], [-1.0], 0.0),
        ])

        new_cons = Constraint(
            '_obj_{}'.format(objective.name),
            new_cons_expr,
            None,
            0.0
        )

        under_result.constraints.append(new_cons)
        return RelaxationResult(new_objective, under_result.constraints)
