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
from suspect.interval import Interval
from galini.core import SumExpression, LinearExpression, QuadraticExpression
from galini.expression_relaxation import ExpressionRelaxation, ExpressionRelaxationResult


class AlphaBBExpressionRelaxation(ExpressionRelaxation):
    def can_relax(self, problem, expr, ctx):
        return True

    def relax(self, problem, expr, ctx):
        alpha = self.compute_alpha(problem, expr, ctx)
        quadratic_exprs = self._quadratic_sum(problem, expr, ctx, alpha)
        children = quadratic_exprs + [expr]
        under_expr = SumExpression(children)
        return ExpressionRelaxationResult(under_expr)

    def compute_alpha(self, problem, expr, ctx):
        xs = self._collect_expr_variables(expr)
        x_bounds = []
        for x in xs:
            x_view = problem.variable_view(x)
            x_l = x_view.lower_bound()
            x_u = x_view.upper_bound()
            if x_l is None or x_u is None:
                raise ValueError('Variables must be bounded: name={}, lb={}, ub={}'.format(
                    x.name, x_l, x_u
                ))
            x_bounds.append(Interval(x_l, x_u))
        tree_data = expr.expression_tree_data()
        f = tree_data.eval(x_bounds)
        H = f.hessian(x_bounds, [1.0])
        n = len(xs)
        min_ = 0
        H = [Interval.zero() + i for i in H]
        for i in range(n):
            h = H[i*n:(i+1)*n]
            v = h[i].lower_bound - sum(max(abs(h[j].lower_bound), abs(h[j].upper_bound)) for j in range(n) if j != i)
            if v < min_:
                min_ = v
        alpha = max(0, -0.5 * min_)
        return alpha

    def _quadratic_sum(self, problem, expr, ctx, alpha):
        xs = self._collect_expr_variables(expr)
        quadratics = []
        linears = []

        for x in xs:
            x_view = problem.variable_view(x)
            x_l = x_view.lower_bound()
            x_u = x_view.upper_bound()
            if x_l is None or x_u is None:
                raise ValueError('Variables must be bounded: name={}, lb={}, ub={}'.format(
                    x.name, x_l, x_u
                ))

            # alpha * x*x
            quadratic_expr = QuadraticExpression([x], [x], [alpha])
            quadratics.append(quadratic_expr)

            # -alpha * (x_l + x_u) * x + alpha * x_l * x_u
            linear_expr = LinearExpression([x], [-alpha*(x_l + x_u)], alpha*x_l*x_u)
            linears.append(linear_expr)
        return [QuadraticExpression(quadratics), LinearExpression(linears)]

    def _collect_expr_variables(self, expr):
        xs = []
        to_visit = [expr]
        seen = set()
        while len(to_visit) > 0:
            curr = to_visit.pop()
            # fra: we use idx as idx of variable for now. Does not work
            # if problem is None
            assert curr.problem is not None
            # TODO(fra): consider case when variable is fixed
            if curr.is_variable():
                xs.append(curr)
            for ch in curr.children:
                if ch.idx not in seen:
                    to_visit.append(ch)
                    seen.add(ch.idx)
        return sorted(xs, key=lambda x: x.idx)
