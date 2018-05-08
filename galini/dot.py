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
"""Output GALINI problems to dat files."""
import pydot

import galini.core as core



def dag_to_pydot_graph(dag, ctx=None):
    """Create pydot graph representing `dag`.

    Arguments
    ---------
    dag: galini.core.Problem
        input problem

    Returns
    -------
        pydot.Dot
    """
    dot = pydot.Dot(rankdir='BT')
    nodes = {}
    # first add variables...
    subrank = pydot.Subgraph(rank='same')
    print(ctx)
    for name, variable in dag.variables.items():
        label = '\n'.join([name, str(variable.idx)])
        if ctx is not None:
            label += '\n' + str(ctx[variable.idx])
        node = pydot.Node(variable.idx, label=label)
        subrank.add_node(node)
    # ... then nodes ..
    dot.add_subgraph(subrank)
    subrank = pydot.Subgraph(rank='same')
    old_depth = 0
    for vertex in dag.sorted_vertices():
        if isinstance(vertex, core.Variable):
            continue
        label = '\n'.join([
            _node_label(vertex),
            str(vertex.idx),
        ])
        if ctx is not None:
            label += '\n' + str(ctx[vertex.idx])
        node = pydot.Node(vertex.idx, label=label)
        nodes[vertex] = node
        vertex_depth = dag.vertex_depth(vertex.idx)
        assert vertex_depth >= old_depth
        if vertex_depth > old_depth:
            dot.add_subgraph(subrank)
            subrank = pydot.Subgraph(rank='same')
            old_depth = vertex_depth
        subrank.add_node(node)
    dot.add_subgraph(subrank)

    # ... then edges ...
    for vertex in dag.sorted_vertices():
        for child in dag.children(vertex):
            edge = pydot.Edge(child.idx, vertex.idx)
            dot.add_edge(edge)
    dot.add_subgraph(subrank)

    # ... finally constraints and objectives
    subrank = pydot.Subgraph(rank='same')
    for name, expr in dag.constraints.items():
        node = pydot.Node(id(expr), label=name, shape='box')
        subrank.add_node(node)
        edge = pydot.Edge(expr.root_expr.idx, id(expr))
        dot.add_edge(edge)
    for name, expr in dag.objectives.items():
        node = pydot.Node(id(expr), label=name, shape='box')
        subrank.add_node(node)
        edge = pydot.Edge(expr.root_expr.idx, id(expr))
        dot.add_edge(edge)

    dot.add_subgraph(subrank)
    return dot


def _node_label(vertex):
    if isinstance(vertex, core.Constant):
        if int(vertex.value) == vertex.value:
            return '{:.0f}'.format(vertex.value)
        return '{:.3f}'.format(vertex.value)
    cls_label = {
        core.ProductExpression: '*',
        core.DivisionExpression: '/',
        core.SumExpression: '+',
        core.PowExpression: '^',
        core.LinearExpression: 'Σ',
        core.NegationExpression: '-',
        core.AbsExpression: '|.|',
        core.SqrtExpression: '√',
        core.ExpExpression: 'exp',
        core.LogExpression: 'log',
        core.SinExpression: 'sin',
        core.CosExpression: 'cos',
        core.TanExpression: 'tan',
        core.AsinExpression: 'asin',
        core.AcosExpression: 'acos',
        core.AtanExpression: 'atan',
    }
    return cls_label.get(
        type(vertex),
        str(type(vertex))
    )
