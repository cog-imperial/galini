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
import networkx as nx
import numpy as np
from suspect.convexity.rules import QuadraticRule
from suspect.expression import ExpressionType
from suspect.fbbt.propagation.rules import (
    QuadraticRule as QuadraticBoundPropagationRule
)

from galini.core import (
    QuadraticExpression,
    LinearExpression,
    SumExpression,
    Constraint,
    Variable,
    Domain,
    ExpressionReference,
)
from galini.underestimators.bilinear import McCormickUnderestimator
from galini.underestimators.underestimator import Underestimator, \
    UnderestimatorResult


class DisaggregateBilinearUnderestimator(Underestimator):
    def __init__(self):
        super().__init__()
        self._call_count = 0
        self._quadratic_rule = QuadraticRule()
        self._quadratic_bound_propagation_rule = \
            QuadraticBoundPropagationRule()
        self._bilinear_underestimator = McCormickUnderestimator(linear=True)

    def can_underestimate(self, problem, expr, ctx):
        return expr.expression_type == ExpressionType.Quadratic

    def underestimate(self, problem, expr, ctx, **kwargs):
        assert expr.expression_type == ExpressionType.Quadratic

        side = kwargs.pop('side')

        # Expression is an equality, use convex envelope
        if side == 2:
            return self._bilinear_underestimator.underestimate(problem, expr, ctx, **kwargs)

        term_graph = nx.Graph()
        term_graph.add_nodes_from(ch.idx for ch in expr.children)
        term_graph.add_edges_from(
            (t.var1.idx, t.var2.idx, {'coefficient': t.coefficient})
            for t in expr.terms
        )

        # Check convexity of each connected subgraph
        convex_exprs = []
        nonconvex_exprs = []
        for connected_component in nx.connected_components(term_graph):
            connected_graph =term_graph.subgraph(connected_component)
            vars1 = []
            vars2 = []
            coefs = []
            for (idx1, idx2) in connected_graph.edges:
                coef = connected_graph.edges[idx1, idx2]['coefficient']
                v1 = problem.variable(idx1)
                v2 = problem.variable(idx2)
                vars1.append(v1)
                vars2.append(v2)
                coefs.append(coef)
            quadratic_expr = QuadraticExpression(vars1, vars2, coefs)
            cvx = self._quadratic_rule.apply(
                quadratic_expr, ctx.convexity, ctx.monotonicity, ctx.bounds
            )
            if cvx.is_convex() and side == 0:
                convex_exprs.append(quadratic_expr)
            elif cvx.is_concave() and side == 1:
                convex_exprs.append(quadratic_expr)
            else:
                nonconvex_exprs.append(quadratic_expr)

        aux_vars = []
        constraints = []
        for quadratic_expr in convex_exprs:
            quadratic_expr_bounds = \
                self._quadratic_bound_propagation_rule.apply(
                    quadratic_expr, ctx.bounds
                )

            aux_w = Variable(
                '_aux_{}'.format(self._call_count),
                quadratic_expr_bounds.lower_bound,
                quadratic_expr_bounds.upper_bound,
                Domain.REAL,
            )

            aux_w.reference = ExpressionReference(quadratic_expr)
            aux_vars.append(aux_w)

            if side == 0:
                lower_bound = None
                upper_bound = 0.0
            elif side == 1:
                lower_bound = 0.0
                upper_bound = None
            else:
                raise RuntimeError('Should not generate disaggregate bilinear terms for equality constraints')

            constraint = Constraint(
                '_disaggregate_aux_{}'.format(self._call_count),
                SumExpression([
                    LinearExpression([aux_w], [-1.0], 0.0),
                    quadratic_expr,
                ]),
                lower_bound,
                upper_bound,
            )
            constraints.append(constraint)
            self._call_count += 1

        nonconvex_quadratic_expr = QuadraticExpression(nonconvex_exprs)
        nonconvex_quadratic_under = \
            self._bilinear_underestimator.underestimate(
                problem, nonconvex_quadratic_expr, ctx, **kwargs
            )
        assert nonconvex_quadratic_under is not None

        aux_vars_expr = LinearExpression(
            aux_vars,
            np.ones_like(aux_vars),
            0.0,
        )

        new_expr = LinearExpression(
            [aux_vars_expr, nonconvex_quadratic_under.expression]
        )

        constraints.extend(nonconvex_quadratic_under.constraints)

        return UnderestimatorResult(new_expr, constraints)
