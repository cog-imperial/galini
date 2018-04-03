"""GALINI dot subcommand."""
import pydot
from galini.subcommands import CliCommand
from galini.pyomo import read_pyomo_model, dag_from_pyomo_model
import galini.dag.expressions as dex


def _node_label(vertex):
    if isinstance(vertex, (dex.Variable, dex.Constraint, dex.Objective)):
        return vertex.name
    elif isinstance(vertex, dex.Constant):
        if int(vertex.value) == vertex.value:
            return '{:.0f}'.format(vertex.value)
        return '{:.3f}'.format(vertex.value)
    cls_label = {
        dex.ProductExpression: '*',
        dex.DivisionExpression: '/',
        dex.SumExpression: '+',
        dex.PowExpression: '^',
        dex.LinearExpression: 'Σ',
        dex.NegationExpression: '-',
        dex.AbsExpression: '|.|',
        dex.SqrtExpression: '√',
        dex.ExpExpression: 'exp',
        dex.LogExpression: 'log',
        dex.SinExpression: 'sin',
        dex.CosExpression: 'cos',
        dex.TanExpression: 'tan',
        dex.AsinExpression: 'asin',
        dex.AcosExpression: 'acos',
        dex.AtanExpression: 'atan',
    }
    return cls_label.get(type(vertex), type(vertex))

def _dag_to_pydot_graph(dag):
    dot = pydot.Dot(rankdir='BT')
    nodes = {}
    # first add nodes...
    subrank = pydot.Subgraph(rank='same')
    old_depth = 0
    for vertex in dag.vertices:
        label = _node_label(vertex)
        node = pydot.Node(id(vertex), label=label)
        if isinstance(vertex, (dex.Constraint, dex.Objective)):
            node.set_shape('box')
        nodes[vertex] = node
        assert vertex.depth >= old_depth
        if vertex.depth > old_depth:
            dot.add_subgraph(subrank)
            subrank = pydot.Subgraph(rank='same')
            old_depth = vertex.depth
        subrank.add_node(node)
    dot.add_subgraph(subrank)
    # ... then edges
    for vertex in dag.vertices:
        for i, child in enumerate(vertex.children):
            to = nodes[vertex]
            from_ = nodes[child]
            edge = pydot.Edge(from_, to, taillabel=str(i))
            dot.add_edge(edge)
    return dot


class DotCommand(CliCommand):
    """Command to output Graphiviz dot file of the problem."""
    def execute(self, args):
        assert args.problem
        pyomo_model = read_pyomo_model(args.problem)
        dag = dag_from_pyomo_model(pyomo_model)
        graph = _dag_to_pydot_graph(dag)

        if args.out:
            graph.write(args.out)
        else:
            print(graph.to_string())

    def help_message(self):
        return "Save GALINI DAG of the problem as Graphviz Dot file"

    def add_parser_arguments(self, parser):
        parser.add_argument('problem')
        parser.add_argument('out', nargs='?')
