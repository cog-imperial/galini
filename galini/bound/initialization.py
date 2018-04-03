import galini.dag.expressions as dex
from galini.dag.visitor import BackwardVisitor
from galini.context import SpecialStructurePropagationContext
from galini.bound import ArbitraryPrecisionBound as Bound


def initialize_bounds(dag):
    """Initialize problem bounds using function domains as bound.

    Parameters
    ----------
    dag: ProblemDag
       the DAG representation of the optimization problem

    Returns
    -------
    ctx: SpecialStructurePropagationContex
       the initial context
    """
    visitor = BoundsInitializationVisitor()
    ctx = SpecialStructurePropagationContext({})
    dag.backward_visit(visitor, ctx)
    return ctx


class BoundsInitializationVisitor(BackwardVisitor):
    def register_handlers(self):
        return {
            dex.Constraint: self.visit_constraint,
            dex.SqrtExpression: self.visit_sqrt,
            dex.LogExpression: self.visit_log,
            dex.AsinExpression: self.visit_asin,
            dex.AcosExpression: self.visit_acos,
            dex.Expression: self.visit_expression,
        }

    def handle_result(self, expr, value, ctx):
        if value is None:
            new_bound = Bound(None, None)
            has_changed = True
        else:
            new_bound = value
            has_changed = True
        ctx.bound[expr] = new_bound
        return has_changed

    def visit_constraint(self, expr, ctx):
        child = expr.children[0]
        bound = Bound(expr.lower_bound, expr.upper_bound)
        return {
            child: bound
        }

    def visit_sqrt(self, expr, _ctx):
        child = expr.children[0]
        return {
            child: Bound(0, None),
        }

    def visit_log(self, expr, _ctx):
        child = expr.children[0]
        return {
            child: Bound(0, None)
        }

    def visit_asin(self, expr, _ctx):
        child = expr.children[0]
        return {
            child: Bound(-1, 1)
        }

    def visit_acos(self, expr, _ctx):
        child = expr.children[0]
        return {
            child: Bound(-1, 1)
        }

    def visit_expression(self, expr, _ctx):
        result = {}
        for child in expr.children:
            result[child] = Bound(None, None)
        return result
