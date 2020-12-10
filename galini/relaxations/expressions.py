# Copyright 2020 Francesco Ceccon
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

"""Functions to relax nonlinear pyomo expresions."""

import networkx as nx
from coramin.utils.coramin_enums import RelaxationSide
from galini.relaxations.multivariate import FactorableConvexExpressionRelaxation
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
from suspect.convexity.rules.quadratic import QuadraticRule
from suspect.pyomo.quadratic import QuadraticExpression

_convexity_rule = QuadraticRule()


def _relax_leaf_to_root_QuadraticExpression(node, values, aux_var_map, degree_map, parent_block,
                                            relaxation_side_map, counter):

    relaxation_side = relaxation_side_map[node]

    term_graph = nx.Graph()
    term_graph.add_edges_from(
        (id(term.var1), id(term.var2), {'term': term})
        for term in node.terms
    )

    disaggregated_exprs = []
    aux_sum = 0.0

    for connected_component in nx.connected_components(term_graph):
        connected_graph = term_graph.subgraph(connected_component)
        expr = 0.0
        aux_var_expr = 0.0
        bilinear_count = 0
        for _, _, data in connected_graph.edges(data=True):
            bilinear_count += 1
            term = data['term']
            var1_id = id(term.var1)
            var2_id = id(term.var2)
            if term.var1 is term.var2:
                aux_var, _ = aux_var_map[var1_id, 'quadratic']
            else:
                aux_var, _ = aux_var_map.get((var1_id, var2_id, 'mul'), (None, None))
                if aux_var is None:
                    aux_var, _ = aux_var_map.get((var2_id, var1_id, 'mul'), (None, None))
                    assert aux_var is not None
            expr += term.coefficient * term.var1 * term.var2
            aux_var_expr += term.coefficient * aux_var

        if bilinear_count == 1:
            aux_sum += aux_var_expr
            continue

        expr = QuadraticExpression(expr)

        cvx = _convexity_rule.apply(expr, None, None, None)

        if relaxation_side == RelaxationSide.UNDER and cvx.is_convex():
            disaggregated_exprs.append((expr, aux_var_expr))
        elif relaxation_side == RelaxationSide.OVER and cvx.is_concave():
            disaggregated_exprs.append((expr, aux_var_expr))
        elif relaxation_side == RelaxationSide.BOTH and cvx.is_convex():
            disaggregated_exprs.append((expr, aux_var_expr))
        else:
            aux_sum += aux_var_expr

    if len(disaggregated_exprs) == 0:
        res = sum(values)
        degree_map[res] = max(degree_map[arg] for arg in values)
        return res

    # replace convex nonlinear expressions with an auxiliary variable
    res = aux_sum
    for expr, aux_var_expr in disaggregated_exprs:
        lb, ub = compute_bounds_on_expr(aux_var_expr)
        new_aux_var = parent_block.aux_vars.add()
        new_aux_var.setlb(lb)
        new_aux_var.setub(ub)
        new_aux_var.value = 0.0
        degree_map[new_aux_var] = 1.0
        relaxation = FactorableConvexExpressionRelaxation()
        relaxation.build(
            aux_var=new_aux_var,
            relaxed_f_x_expr=aux_var_expr,
            original_f_x_expr=expr,
        )
        setattr(parent_block.relaxations, 'rel'+str(counter), relaxation)
        counter.increment()
        res += new_aux_var
    degree_map[res] = 1.0
    return res
